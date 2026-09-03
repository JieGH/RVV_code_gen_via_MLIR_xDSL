#include "kernels_RVV_8x20_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_1x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 1] @DRAM
// )
void gemm_RVV_1x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
}

// gemm_RVV_1x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 1] @DRAM
// )
void gemm_RVV_1x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
}

// gemm_RVV_1x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 1] @DRAM
// )
void gemm_RVV_1x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
}

// gemm_RVV_1x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 1] @DRAM
// )
void gemm_RVV_1x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
}

// gemm_RVV_1x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 1] @DRAM
// )
void gemm_RVV_1x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
}

// gemm_RVV_1x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 1] @DRAM
// )
void gemm_RVV_1x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
}

// gemm_RVV_1x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 1] @DRAM
// )
void gemm_RVV_1x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
}

// gemm_RVV_1x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 1] @DRAM
// )
void gemm_RVV_1x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
}

// gemm_RVV_1x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 1] @DRAM
// )
void gemm_RVV_1x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
}

// gemm_RVV_1x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 1] @DRAM
// )
void gemm_RVV_1x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
}

// gemm_RVV_1x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 1] @DRAM
// )
void gemm_RVV_1x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
}

// gemm_RVV_1x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 1] @DRAM
// )
void gemm_RVV_1x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
}

// gemm_RVV_1x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 1] @DRAM
// )
void gemm_RVV_1x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
}

// gemm_RVV_1x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 1] @DRAM
// )
void gemm_RVV_1x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(1));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
}

// gemm_RVV_1x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 1] @DRAM
// )
void gemm_RVV_1x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(1));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
}

// gemm_RVV_1x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 1] @DRAM
// )
void gemm_RVV_1x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(1));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(1));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(1));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
}

// gemm_RVV_1x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 1] @DRAM
// )
void gemm_RVV_1x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(1));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(1));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
}

// gemm_RVV_1x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 1] @DRAM
// )
void gemm_RVV_1x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(1));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(1));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(1));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(1));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(1));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
}

// gemm_RVV_1x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 1] @DRAM
// )
void gemm_RVV_1x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(1));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(1));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(1));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(1));
}

// gemm_RVV_1x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 1] @DRAM
// )
void gemm_RVV_1x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(1));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(1));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(1));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(1));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(1));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(1));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(1));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(1));
}

// gemm_RVV_1x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 1] @DRAM
// )
void gemm_RVV_1x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_19 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(1));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(1));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(1));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(1));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(1));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(1));
}

// gemm_RVV_1x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 1] @DRAM
// )
void gemm_RVV_1x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(1));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(1));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(1));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(1));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(1));
C_reg_19 = __riscv_vle16_v_f16m1(&C.data[(19) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(1));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(1));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(1));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(1));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(1));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(1));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(1));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(1));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(1));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(1));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(1));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(1));
}

// gemm_RVV_1x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_1x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_1x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RVV_1x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
}

// gemm_RVV_1x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RVV_1x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
}

// gemm_RVV_1x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RVV_1x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
}

// gemm_RVV_1x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RVV_1x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
}

// gemm_RVV_1x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RVV_1x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
}

// gemm_RVV_1x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RVV_1x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
}

// gemm_RVV_1x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RVV_1x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
}

// gemm_RVV_1x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RVV_1x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
}

// gemm_RVV_1x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RVV_1x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
}

// gemm_RVV_1x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RVV_1x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
}

// gemm_RVV_1x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RVV_1x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
}

// gemm_RVV_1x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RVV_1x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
}

// gemm_RVV_1x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 1] @DRAM
// )
void gemm_RVV_1x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
}

// gemm_RVV_1x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 1] @DRAM
// )
void gemm_RVV_1x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(1));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
}

// gemm_RVV_2x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 2] @DRAM
// )
void gemm_RVV_2x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
}

// gemm_RVV_2x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 2] @DRAM
// )
void gemm_RVV_2x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
}

// gemm_RVV_2x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 2] @DRAM
// )
void gemm_RVV_2x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
}

// gemm_RVV_2x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 2] @DRAM
// )
void gemm_RVV_2x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
}

// gemm_RVV_2x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 2] @DRAM
// )
void gemm_RVV_2x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
}

// gemm_RVV_2x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 2] @DRAM
// )
void gemm_RVV_2x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
}

// gemm_RVV_2x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 2] @DRAM
// )
void gemm_RVV_2x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
}

// gemm_RVV_2x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 2] @DRAM
// )
void gemm_RVV_2x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
}

// gemm_RVV_2x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 2] @DRAM
// )
void gemm_RVV_2x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
}

// gemm_RVV_2x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 2] @DRAM
// )
void gemm_RVV_2x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
}

// gemm_RVV_2x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 2] @DRAM
// )
void gemm_RVV_2x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
}

// gemm_RVV_2x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 2] @DRAM
// )
void gemm_RVV_2x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
}

// gemm_RVV_2x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 2] @DRAM
// )
void gemm_RVV_2x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
}

// gemm_RVV_2x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 2] @DRAM
// )
void gemm_RVV_2x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(2));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
}

// gemm_RVV_2x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 2] @DRAM
// )
void gemm_RVV_2x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(2));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
}

// gemm_RVV_2x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 2] @DRAM
// )
void gemm_RVV_2x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(2));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(2));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(2));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
}

// gemm_RVV_2x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 2] @DRAM
// )
void gemm_RVV_2x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(2));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(2));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
}

// gemm_RVV_2x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 2] @DRAM
// )
void gemm_RVV_2x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(2));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(2));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(2));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(2));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(2));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
}

// gemm_RVV_2x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 2] @DRAM
// )
void gemm_RVV_2x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(2));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(2));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(2));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(2));
}

// gemm_RVV_2x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 2] @DRAM
// )
void gemm_RVV_2x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(2));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(2));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(2));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(2));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(2));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(2));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(2));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(2));
}

// gemm_RVV_2x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 2] @DRAM
// )
void gemm_RVV_2x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_19 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(2));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(2));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(2));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(2));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(2));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(2));
}

// gemm_RVV_2x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 2] @DRAM
// )
void gemm_RVV_2x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(2));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(2));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(2));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(2));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(2));
C_reg_19 = __riscv_vle16_v_f16m1(&C.data[(19) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(2));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(2));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(2));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(2));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(2));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(2));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(2));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(2));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(2));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(2));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(2));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(2));
}

// gemm_RVV_2x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_2x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_2x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RVV_2x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
}

// gemm_RVV_2x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RVV_2x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
}

// gemm_RVV_2x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RVV_2x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
}

// gemm_RVV_2x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RVV_2x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
}

// gemm_RVV_2x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RVV_2x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
}

// gemm_RVV_2x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RVV_2x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
}

// gemm_RVV_2x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RVV_2x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
}

// gemm_RVV_2x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RVV_2x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
}

// gemm_RVV_2x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RVV_2x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
}

// gemm_RVV_2x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RVV_2x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
}

// gemm_RVV_2x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RVV_2x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
}

// gemm_RVV_2x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RVV_2x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
}

// gemm_RVV_2x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 2] @DRAM
// )
void gemm_RVV_2x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
}

// gemm_RVV_2x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 2] @DRAM
// )
void gemm_RVV_2x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(2));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
}

// gemm_RVV_3x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 3] @DRAM
// )
void gemm_RVV_3x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
}

// gemm_RVV_3x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 3] @DRAM
// )
void gemm_RVV_3x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
}

// gemm_RVV_3x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 3] @DRAM
// )
void gemm_RVV_3x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
}

// gemm_RVV_3x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 3] @DRAM
// )
void gemm_RVV_3x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
}

// gemm_RVV_3x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 3] @DRAM
// )
void gemm_RVV_3x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
}

// gemm_RVV_3x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 3] @DRAM
// )
void gemm_RVV_3x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
}

// gemm_RVV_3x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 3] @DRAM
// )
void gemm_RVV_3x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
}

// gemm_RVV_3x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 3] @DRAM
// )
void gemm_RVV_3x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
}

// gemm_RVV_3x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 3] @DRAM
// )
void gemm_RVV_3x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
}

// gemm_RVV_3x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 3] @DRAM
// )
void gemm_RVV_3x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
}

// gemm_RVV_3x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 3] @DRAM
// )
void gemm_RVV_3x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
}

// gemm_RVV_3x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 3] @DRAM
// )
void gemm_RVV_3x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
}

// gemm_RVV_3x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 3] @DRAM
// )
void gemm_RVV_3x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
}

// gemm_RVV_3x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 3] @DRAM
// )
void gemm_RVV_3x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(3));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
}

// gemm_RVV_3x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 3] @DRAM
// )
void gemm_RVV_3x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(3));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
}

// gemm_RVV_3x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 3] @DRAM
// )
void gemm_RVV_3x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(3));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(3));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(3));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
}

// gemm_RVV_3x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 3] @DRAM
// )
void gemm_RVV_3x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(3));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(3));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
}

// gemm_RVV_3x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 3] @DRAM
// )
void gemm_RVV_3x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(3));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(3));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(3));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(3));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(3));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
}

// gemm_RVV_3x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 3] @DRAM
// )
void gemm_RVV_3x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(3));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(3));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(3));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(3));
}

// gemm_RVV_3x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 3] @DRAM
// )
void gemm_RVV_3x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(3));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(3));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(3));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(3));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(3));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(3));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(3));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(3));
}

// gemm_RVV_3x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 3] @DRAM
// )
void gemm_RVV_3x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_19 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(3));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(3));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(3));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(3));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(3));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(3));
}

// gemm_RVV_3x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 3] @DRAM
// )
void gemm_RVV_3x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(3));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(3));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(3));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(3));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(3));
C_reg_19 = __riscv_vle16_v_f16m1(&C.data[(19) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(3));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(3));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(3));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(3));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(3));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(3));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(3));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(3));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(3));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(3));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(3));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(3));
}

// gemm_RVV_3x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_3x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_3x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RVV_3x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
}

// gemm_RVV_3x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RVV_3x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
}

// gemm_RVV_3x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RVV_3x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
}

// gemm_RVV_3x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RVV_3x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
}

// gemm_RVV_3x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RVV_3x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
}

// gemm_RVV_3x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RVV_3x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
}

// gemm_RVV_3x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RVV_3x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
}

// gemm_RVV_3x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RVV_3x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
}

// gemm_RVV_3x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RVV_3x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
}

// gemm_RVV_3x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RVV_3x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
}

// gemm_RVV_3x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RVV_3x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
}

// gemm_RVV_3x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RVV_3x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
}

// gemm_RVV_3x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 3] @DRAM
// )
void gemm_RVV_3x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
}

// gemm_RVV_3x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 3] @DRAM
// )
void gemm_RVV_3x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(3));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
}

// gemm_RVV_4x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 4] @DRAM
// )
void gemm_RVV_4x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
}

// gemm_RVV_4x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 4] @DRAM
// )
void gemm_RVV_4x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
}

// gemm_RVV_4x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 4] @DRAM
// )
void gemm_RVV_4x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
}

// gemm_RVV_4x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 4] @DRAM
// )
void gemm_RVV_4x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
}

// gemm_RVV_4x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 4] @DRAM
// )
void gemm_RVV_4x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
}

// gemm_RVV_4x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 4] @DRAM
// )
void gemm_RVV_4x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
}

// gemm_RVV_4x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 4] @DRAM
// )
void gemm_RVV_4x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
}

// gemm_RVV_4x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 4] @DRAM
// )
void gemm_RVV_4x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
}

// gemm_RVV_4x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 4] @DRAM
// )
void gemm_RVV_4x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
}

// gemm_RVV_4x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 4] @DRAM
// )
void gemm_RVV_4x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
}

// gemm_RVV_4x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 4] @DRAM
// )
void gemm_RVV_4x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
}

// gemm_RVV_4x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 4] @DRAM
// )
void gemm_RVV_4x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
}

// gemm_RVV_4x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 4] @DRAM
// )
void gemm_RVV_4x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
}

// gemm_RVV_4x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 4] @DRAM
// )
void gemm_RVV_4x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(4));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
}

// gemm_RVV_4x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 4] @DRAM
// )
void gemm_RVV_4x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(4));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
}

// gemm_RVV_4x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 4] @DRAM
// )
void gemm_RVV_4x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(4));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(4));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(4));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
}

// gemm_RVV_4x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 4] @DRAM
// )
void gemm_RVV_4x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(4));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(4));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
}

// gemm_RVV_4x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 4] @DRAM
// )
void gemm_RVV_4x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(4));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(4));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(4));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(4));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(4));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
}

// gemm_RVV_4x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 4] @DRAM
// )
void gemm_RVV_4x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(4));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(4));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(4));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(4));
}

// gemm_RVV_4x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 4] @DRAM
// )
void gemm_RVV_4x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(4));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(4));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(4));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(4));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(4));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(4));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(4));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(4));
}

// gemm_RVV_4x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 4] @DRAM
// )
void gemm_RVV_4x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_19 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(4));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(4));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(4));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(4));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(4));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(4));
}

// gemm_RVV_4x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 4] @DRAM
// )
void gemm_RVV_4x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(4));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(4));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(4));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(4));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(4));
C_reg_19 = __riscv_vle16_v_f16m1(&C.data[(19) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(4));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(4));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(4));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(4));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(4));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(4));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(4));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(4));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(4));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(4));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(4));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(4));
}

// gemm_RVV_4x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_4x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_4x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RVV_4x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
}

// gemm_RVV_4x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RVV_4x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
}

// gemm_RVV_4x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RVV_4x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
}

// gemm_RVV_4x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RVV_4x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
}

// gemm_RVV_4x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RVV_4x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
}

// gemm_RVV_4x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RVV_4x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
}

// gemm_RVV_4x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RVV_4x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
}

// gemm_RVV_4x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RVV_4x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
}

// gemm_RVV_4x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RVV_4x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
}

// gemm_RVV_4x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RVV_4x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
}

// gemm_RVV_4x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RVV_4x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
}

// gemm_RVV_4x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RVV_4x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
}

// gemm_RVV_4x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 4] @DRAM
// )
void gemm_RVV_4x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
}

// gemm_RVV_4x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 4] @DRAM
// )
void gemm_RVV_4x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(4));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
}

// gemm_RVV_5x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 5] @DRAM
// )
void gemm_RVV_5x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
}

// gemm_RVV_5x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 5] @DRAM
// )
void gemm_RVV_5x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
}

// gemm_RVV_5x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 5] @DRAM
// )
void gemm_RVV_5x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
}

// gemm_RVV_5x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 5] @DRAM
// )
void gemm_RVV_5x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
}

// gemm_RVV_5x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 5] @DRAM
// )
void gemm_RVV_5x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
}

// gemm_RVV_5x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 5] @DRAM
// )
void gemm_RVV_5x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(5));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
}

// gemm_RVV_5x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 5] @DRAM
// )
void gemm_RVV_5x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
}

// gemm_RVV_5x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 5] @DRAM
// )
void gemm_RVV_5x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(5));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(5));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
}

// gemm_RVV_5x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 5] @DRAM
// )
void gemm_RVV_5x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
}

// gemm_RVV_5x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 5] @DRAM
// )
void gemm_RVV_5x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(5));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(5));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(5));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
}

// gemm_RVV_5x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 5] @DRAM
// )
void gemm_RVV_5x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
}

// gemm_RVV_5x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 5] @DRAM
// )
void gemm_RVV_5x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(5));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(5));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(5));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(5));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
}

// gemm_RVV_5x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 5] @DRAM
// )
void gemm_RVV_5x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(5));
}

// gemm_RVV_5x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 5] @DRAM
// )
void gemm_RVV_5x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(5));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(5));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(5));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(5));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(5));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(5));
}

// gemm_RVV_5x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 5] @DRAM
// )
void gemm_RVV_5x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(5));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(5));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(5));
}

// gemm_RVV_5x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 5] @DRAM
// )
void gemm_RVV_5x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(5));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(5));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(5));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(5));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(5));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(5));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(5));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(5));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(5));
}

// gemm_RVV_5x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 5] @DRAM
// )
void gemm_RVV_5x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(5));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(5));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(5));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(5));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(5));
}

// gemm_RVV_5x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 5] @DRAM
// )
void gemm_RVV_5x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(5));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(5));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(5));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(5));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(5));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(5));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(5));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(5));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(5));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(5));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(5));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(5));
}

// gemm_RVV_5x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 5] @DRAM
// )
void gemm_RVV_5x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(5));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(5));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(5));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(5));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(5));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(5));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(5));
}

// gemm_RVV_5x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 5] @DRAM
// )
void gemm_RVV_5x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(5));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(5));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(5));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(5));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(5));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(5));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(5));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(5));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(5));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(5));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(5));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(5));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(5));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(5));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(5));
}

// gemm_RVV_5x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
}

// gemm_RVV_5x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
}

// gemm_RVV_5x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 5] @DRAM
// )
void gemm_RVV_5x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_19 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(5));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(5));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(5));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(5));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(5));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(5));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(5));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(5));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(5));
}

// gemm_RVV_5x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 5] @DRAM
// )
void gemm_RVV_5x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(5));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(5));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(5));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(5));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(5));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(5));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(5));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(5));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(5));
C_reg_19 = __riscv_vle16_v_f16m1(&C.data[(19) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(5));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(5));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(5));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(5));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(5));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(5));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(5));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(5));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(5));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(5));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(5));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(5));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(5));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(5));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(5));
}

// gemm_RVV_5x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
}

// gemm_RVV_5x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
}

// gemm_RVV_5x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RVV_5x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
}

// gemm_RVV_5x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RVV_5x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
}

// gemm_RVV_5x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RVV_5x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
}

// gemm_RVV_5x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RVV_5x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
}

// gemm_RVV_5x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RVV_5x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
}

// gemm_RVV_5x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RVV_5x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
}

// gemm_RVV_5x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RVV_5x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
}

// gemm_RVV_5x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RVV_5x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
}

// gemm_RVV_5x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RVV_5x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
}

// gemm_RVV_5x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RVV_5x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
}

// gemm_RVV_5x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RVV_5x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
}

// gemm_RVV_5x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RVV_5x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
}

// gemm_RVV_5x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 5] @DRAM
// )
void gemm_RVV_5x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
}

// gemm_RVV_5x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 5] @DRAM
// )
void gemm_RVV_5x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(5));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(5));
}

// gemm_RVV_6x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 6] @DRAM
// )
void gemm_RVV_6x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
}

// gemm_RVV_6x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 6] @DRAM
// )
void gemm_RVV_6x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
}

// gemm_RVV_6x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 6] @DRAM
// )
void gemm_RVV_6x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
}

// gemm_RVV_6x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 6] @DRAM
// )
void gemm_RVV_6x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
}

// gemm_RVV_6x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 6] @DRAM
// )
void gemm_RVV_6x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
}

// gemm_RVV_6x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 6] @DRAM
// )
void gemm_RVV_6x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(6));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
}

// gemm_RVV_6x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 6] @DRAM
// )
void gemm_RVV_6x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
}

// gemm_RVV_6x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 6] @DRAM
// )
void gemm_RVV_6x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(6));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(6));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
}

// gemm_RVV_6x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 6] @DRAM
// )
void gemm_RVV_6x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
}

// gemm_RVV_6x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 6] @DRAM
// )
void gemm_RVV_6x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(6));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(6));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(6));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
}

// gemm_RVV_6x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 6] @DRAM
// )
void gemm_RVV_6x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
}

// gemm_RVV_6x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 6] @DRAM
// )
void gemm_RVV_6x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(6));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(6));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(6));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(6));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
}

// gemm_RVV_6x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 6] @DRAM
// )
void gemm_RVV_6x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(6));
}

// gemm_RVV_6x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 6] @DRAM
// )
void gemm_RVV_6x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(6));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(6));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(6));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(6));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(6));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(6));
}

// gemm_RVV_6x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 6] @DRAM
// )
void gemm_RVV_6x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(6));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(6));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(6));
}

// gemm_RVV_6x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 6] @DRAM
// )
void gemm_RVV_6x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(6));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(6));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(6));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(6));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(6));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(6));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(6));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(6));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(6));
}

// gemm_RVV_6x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 6] @DRAM
// )
void gemm_RVV_6x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(6));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(6));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(6));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(6));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(6));
}

// gemm_RVV_6x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 6] @DRAM
// )
void gemm_RVV_6x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(6));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(6));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(6));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(6));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(6));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(6));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(6));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(6));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(6));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(6));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(6));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(6));
}

// gemm_RVV_6x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 6] @DRAM
// )
void gemm_RVV_6x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(6));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(6));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(6));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(6));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(6));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(6));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(6));
}

// gemm_RVV_6x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 6] @DRAM
// )
void gemm_RVV_6x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(6));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(6));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(6));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(6));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(6));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(6));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(6));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(6));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(6));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(6));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(6));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(6));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(6));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(6));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(6));
}

// gemm_RVV_6x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
}

// gemm_RVV_6x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
}

// gemm_RVV_6x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 6] @DRAM
// )
void gemm_RVV_6x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_19 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(6));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(6));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(6));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(6));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(6));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(6));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(6));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(6));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(6));
}

// gemm_RVV_6x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 6] @DRAM
// )
void gemm_RVV_6x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(6));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(6));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(6));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(6));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(6));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(6));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(6));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(6));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(6));
C_reg_19 = __riscv_vle16_v_f16m1(&C.data[(19) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(6));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(6));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(6));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(6));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(6));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(6));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(6));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(6));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(6));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(6));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(6));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(6));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(6));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(6));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(6));
}

// gemm_RVV_6x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
}

// gemm_RVV_6x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
}

// gemm_RVV_6x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RVV_6x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
}

// gemm_RVV_6x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RVV_6x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
}

// gemm_RVV_6x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RVV_6x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
}

// gemm_RVV_6x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RVV_6x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
}

// gemm_RVV_6x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RVV_6x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
}

// gemm_RVV_6x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RVV_6x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
}

// gemm_RVV_6x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RVV_6x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
}

// gemm_RVV_6x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RVV_6x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
}

// gemm_RVV_6x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RVV_6x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
}

// gemm_RVV_6x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RVV_6x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
}

// gemm_RVV_6x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RVV_6x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
}

// gemm_RVV_6x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RVV_6x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
}

// gemm_RVV_6x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 6] @DRAM
// )
void gemm_RVV_6x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
}

// gemm_RVV_6x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 6] @DRAM
// )
void gemm_RVV_6x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(6));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(6));
}

// gemm_RVV_7x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 7] @DRAM
// )
void gemm_RVV_7x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
}

// gemm_RVV_7x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 7] @DRAM
// )
void gemm_RVV_7x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
}

// gemm_RVV_7x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 7] @DRAM
// )
void gemm_RVV_7x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
}

// gemm_RVV_7x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 7] @DRAM
// )
void gemm_RVV_7x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
}

// gemm_RVV_7x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 7] @DRAM
// )
void gemm_RVV_7x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
}

// gemm_RVV_7x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 7] @DRAM
// )
void gemm_RVV_7x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(7));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
}

// gemm_RVV_7x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 7] @DRAM
// )
void gemm_RVV_7x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
}

// gemm_RVV_7x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 7] @DRAM
// )
void gemm_RVV_7x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(7));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(7));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
}

// gemm_RVV_7x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 7] @DRAM
// )
void gemm_RVV_7x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
}

// gemm_RVV_7x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 7] @DRAM
// )
void gemm_RVV_7x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(7));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(7));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(7));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
}

// gemm_RVV_7x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 7] @DRAM
// )
void gemm_RVV_7x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
}

// gemm_RVV_7x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 7] @DRAM
// )
void gemm_RVV_7x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(7));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(7));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(7));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(7));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
}

// gemm_RVV_7x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 7] @DRAM
// )
void gemm_RVV_7x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(7));
}

// gemm_RVV_7x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 7] @DRAM
// )
void gemm_RVV_7x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(7));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(7));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(7));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(7));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(7));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(7));
}

// gemm_RVV_7x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 7] @DRAM
// )
void gemm_RVV_7x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(7));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(7));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(7));
}

// gemm_RVV_7x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 7] @DRAM
// )
void gemm_RVV_7x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(7));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(7));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(7));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(7));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(7));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(7));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(7));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(7));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(7));
}

// gemm_RVV_7x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 7] @DRAM
// )
void gemm_RVV_7x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(7));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(7));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(7));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(7));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(7));
}

// gemm_RVV_7x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 7] @DRAM
// )
void gemm_RVV_7x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(7));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(7));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(7));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(7));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(7));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(7));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(7));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(7));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(7));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(7));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(7));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(7));
}

// gemm_RVV_7x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 7] @DRAM
// )
void gemm_RVV_7x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(7));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(7));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(7));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(7));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(7));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(7));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(7));
}

// gemm_RVV_7x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 7] @DRAM
// )
void gemm_RVV_7x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(7));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(7));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(7));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(7));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(7));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(7));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(7));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(7));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(7));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(7));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(7));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(7));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(7));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(7));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(7));
}

// gemm_RVV_7x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
}

// gemm_RVV_7x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
}

// gemm_RVV_7x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 7] @DRAM
// )
void gemm_RVV_7x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_19 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(7));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(7));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(7));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(7));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(7));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(7));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(7));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(7));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(7));
}

// gemm_RVV_7x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 7] @DRAM
// )
void gemm_RVV_7x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(7));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(7));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(7));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(7));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(7));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(7));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(7));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(7));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(7));
C_reg_19 = __riscv_vle16_v_f16m1(&C.data[(19) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(7));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(7));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(7));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(7));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(7));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(7));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(7));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(7));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(7));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(7));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(7));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(7));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(7));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(7));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(7));
}

// gemm_RVV_7x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
}

// gemm_RVV_7x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
}

// gemm_RVV_7x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RVV_7x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
}

// gemm_RVV_7x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RVV_7x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
}

// gemm_RVV_7x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RVV_7x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
}

// gemm_RVV_7x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RVV_7x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
}

// gemm_RVV_7x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RVV_7x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
}

// gemm_RVV_7x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RVV_7x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
}

// gemm_RVV_7x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RVV_7x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
}

// gemm_RVV_7x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RVV_7x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
}

// gemm_RVV_7x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RVV_7x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
}

// gemm_RVV_7x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RVV_7x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
}

// gemm_RVV_7x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RVV_7x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
}

// gemm_RVV_7x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RVV_7x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
}

// gemm_RVV_7x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 7] @DRAM
// )
void gemm_RVV_7x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
}

// gemm_RVV_7x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 7] @DRAM
// )
void gemm_RVV_7x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(7));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(7));
}

// gemm_RVV_8x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 8] @DRAM
// )
void gemm_RVV_8x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
}

// gemm_RVV_8x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 8] @DRAM
// )
void gemm_RVV_8x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
}

// gemm_RVV_8x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 8] @DRAM
// )
void gemm_RVV_8x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
}

// gemm_RVV_8x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 8] @DRAM
// )
void gemm_RVV_8x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
}

// gemm_RVV_8x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 8] @DRAM
// )
void gemm_RVV_8x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
}

// gemm_RVV_8x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 8] @DRAM
// )
void gemm_RVV_8x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(8));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
}

// gemm_RVV_8x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 8] @DRAM
// )
void gemm_RVV_8x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
}

// gemm_RVV_8x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 8] @DRAM
// )
void gemm_RVV_8x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(8));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(8));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
}

// gemm_RVV_8x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 8] @DRAM
// )
void gemm_RVV_8x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
}

// gemm_RVV_8x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 8] @DRAM
// )
void gemm_RVV_8x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(8));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(8));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(8));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
}

// gemm_RVV_8x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 8] @DRAM
// )
void gemm_RVV_8x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
}

// gemm_RVV_8x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 8] @DRAM
// )
void gemm_RVV_8x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(8));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(8));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(8));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(8));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
}

// gemm_RVV_8x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 8] @DRAM
// )
void gemm_RVV_8x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(8));
}

// gemm_RVV_8x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 8] @DRAM
// )
void gemm_RVV_8x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(8));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(8));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(8));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(8));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(8));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(8));
}

// gemm_RVV_8x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 8] @DRAM
// )
void gemm_RVV_8x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(8));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(8));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(8));
}

// gemm_RVV_8x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 8] @DRAM
// )
void gemm_RVV_8x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(8));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(8));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(8));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(8));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(8));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(8));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(8));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(8));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(8));
}

// gemm_RVV_8x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 8] @DRAM
// )
void gemm_RVV_8x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(8));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(8));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(8));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(8));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(8));
}

// gemm_RVV_8x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 8] @DRAM
// )
void gemm_RVV_8x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(8));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(8));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(8));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(8));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(8));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(8));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(8));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(8));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(8));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(8));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(8));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(8));
}

// gemm_RVV_8x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 8] @DRAM
// )
void gemm_RVV_8x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(8));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(8));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(8));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(8));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(8));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(8));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(8));
}

// gemm_RVV_8x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 8] @DRAM
// )
void gemm_RVV_8x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(8));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(8));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(8));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(8));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(8));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(8));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(8));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(8));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(8));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(8));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(8));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(8));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(8));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(8));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(8));
}

// gemm_RVV_8x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
}

// gemm_RVV_8x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
}

// gemm_RVV_8x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 8] @DRAM
// )
void gemm_RVV_8x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_16 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_17 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_18 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_19 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(8));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(8));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(8));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(8));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(8));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(8));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(8));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(8));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(8));
}

// gemm_RVV_8x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 8] @DRAM
// )
void gemm_RVV_8x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
vfloat16m1_t C_reg_16;
vfloat16m1_t C_reg_17;
vfloat16m1_t C_reg_18;
vfloat16m1_t C_reg_19;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(8));
C_reg_11 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(8));
C_reg_12 = __riscv_vle16_v_f16m1(&C.data[(12) * (C.strides[0])],(8));
C_reg_13 = __riscv_vle16_v_f16m1(&C.data[(13) * (C.strides[0])],(8));
C_reg_14 = __riscv_vle16_v_f16m1(&C.data[(14) * (C.strides[0])],(8));
C_reg_15 = __riscv_vle16_v_f16m1(&C.data[(15) * (C.strides[0])],(8));
C_reg_16 = __riscv_vle16_v_f16m1(&C.data[(16) * (C.strides[0])],(8));
C_reg_17 = __riscv_vle16_v_f16m1(&C.data[(17) * (C.strides[0])],(8));
C_reg_18 = __riscv_vle16_v_f16m1(&C.data[(18) * (C.strides[0])],(8));
C_reg_19 = __riscv_vle16_v_f16m1(&C.data[(19) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
  C_reg_9 = __riscv_vfmacc_vf_f16m1(C_reg_9, B.data[(k) * (B.strides[0]) + 9], A_reg,(8));
  C_reg_10 = __riscv_vfmacc_vf_f16m1(C_reg_10, B.data[(k) * (B.strides[0]) + 10], A_reg,(8));
  C_reg_11 = __riscv_vfmacc_vf_f16m1(C_reg_11, B.data[(k) * (B.strides[0]) + 11], A_reg,(8));
  C_reg_12 = __riscv_vfmacc_vf_f16m1(C_reg_12, B.data[(k) * (B.strides[0]) + 12], A_reg,(8));
  C_reg_13 = __riscv_vfmacc_vf_f16m1(C_reg_13, B.data[(k) * (B.strides[0]) + 13], A_reg,(8));
  C_reg_14 = __riscv_vfmacc_vf_f16m1(C_reg_14, B.data[(k) * (B.strides[0]) + 14], A_reg,(8));
  C_reg_15 = __riscv_vfmacc_vf_f16m1(C_reg_15, B.data[(k) * (B.strides[0]) + 15], A_reg,(8));
  C_reg_16 = __riscv_vfmacc_vf_f16m1(C_reg_16, B.data[(k) * (B.strides[0]) + 16], A_reg,(8));
  C_reg_17 = __riscv_vfmacc_vf_f16m1(C_reg_17, B.data[(k) * (B.strides[0]) + 17], A_reg,(8));
  C_reg_18 = __riscv_vfmacc_vf_f16m1(C_reg_18, B.data[(k) * (B.strides[0]) + 18], A_reg,(8));
  C_reg_19 = __riscv_vfmacc_vf_f16m1(C_reg_19, B.data[(k) * (B.strides[0]) + 19], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C.data[(12) * (C.strides[0])], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C.data[(13) * (C.strides[0])], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C.data[(14) * (C.strides[0])], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C.data[(15) * (C.strides[0])], C_reg_15,(8));
__riscv_vse16_v_f16m1(&C.data[(16) * (C.strides[0])], C_reg_16,(8));
__riscv_vse16_v_f16m1(&C.data[(17) * (C.strides[0])], C_reg_17,(8));
__riscv_vse16_v_f16m1(&C.data[(18) * (C.strides[0])], C_reg_18,(8));
__riscv_vse16_v_f16m1(&C.data[(19) * (C.strides[0])], C_reg_19,(8));
}

// gemm_RVV_8x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
}

// gemm_RVV_8x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
}

// gemm_RVV_8x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RVV_8x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
}

// gemm_RVV_8x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RVV_8x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
}

// gemm_RVV_8x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RVV_8x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
}

// gemm_RVV_8x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RVV_8x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
}

// gemm_RVV_8x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RVV_8x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
}

// gemm_RVV_8x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RVV_8x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
}

// gemm_RVV_8x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RVV_8x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
}

// gemm_RVV_8x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RVV_8x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
}

// gemm_RVV_8x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RVV_8x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
}

// gemm_RVV_8x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RVV_8x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
}

// gemm_RVV_8x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RVV_8x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
}

// gemm_RVV_8x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RVV_8x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
}

// gemm_RVV_8x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 8] @DRAM
// )
void gemm_RVV_8x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
}

// gemm_RVV_8x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 8] @DRAM
// )
void gemm_RVV_8x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_8;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B.data[(k) * (B.strides[0]) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B.data[(k) * (B.strides[0]) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B.data[(k) * (B.strides[0]) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B.data[(k) * (B.strides[0]) + 7], A_reg,(8));
  C_reg_8 = __riscv_vfmacc_vf_f16m1(C_reg_8, B.data[(k) * (B.strides[0]) + 8], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8,(8));
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
