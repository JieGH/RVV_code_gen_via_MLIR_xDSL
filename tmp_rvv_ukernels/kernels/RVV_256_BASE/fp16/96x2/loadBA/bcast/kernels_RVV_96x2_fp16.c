#include "kernels_RVV_96x2_fp16.h"



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
void gemm_RVV_10x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(10));
}

// gemm_RVV_10x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 10] @DRAM
// )
void gemm_RVV_10x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(10));
}

// gemm_RVV_10x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 10] @DRAM
// )
void gemm_RVV_10x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(10));
}

// gemm_RVV_10x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 10] @DRAM
// )
void gemm_RVV_10x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(10));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(10));
}

// gemm_RVV_11x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 11] @DRAM
// )
void gemm_RVV_11x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(11));
}

// gemm_RVV_11x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 11] @DRAM
// )
void gemm_RVV_11x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(11));
}

// gemm_RVV_11x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 11] @DRAM
// )
void gemm_RVV_11x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(11));
}

// gemm_RVV_11x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 11] @DRAM
// )
void gemm_RVV_11x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(11));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(11));
}

// gemm_RVV_12x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 12] @DRAM
// )
void gemm_RVV_12x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(12));
}

// gemm_RVV_12x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 12] @DRAM
// )
void gemm_RVV_12x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(12));
}

// gemm_RVV_12x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 12] @DRAM
// )
void gemm_RVV_12x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(12));
}

// gemm_RVV_12x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 12] @DRAM
// )
void gemm_RVV_12x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(12));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(12));
}

// gemm_RVV_13x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 13] @DRAM
// )
void gemm_RVV_13x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(13));
}

// gemm_RVV_13x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 13] @DRAM
// )
void gemm_RVV_13x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(13));
}

// gemm_RVV_13x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 13] @DRAM
// )
void gemm_RVV_13x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(13));
}

// gemm_RVV_13x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 13] @DRAM
// )
void gemm_RVV_13x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(13));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(13));
}

// gemm_RVV_14x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 14] @DRAM
// )
void gemm_RVV_14x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(14));
}

// gemm_RVV_14x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 14] @DRAM
// )
void gemm_RVV_14x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(14));
}

// gemm_RVV_14x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 14] @DRAM
// )
void gemm_RVV_14x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(14));
}

// gemm_RVV_14x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 14] @DRAM
// )
void gemm_RVV_14x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(14));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(14));
}

// gemm_RVV_15x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 15] @DRAM
// )
void gemm_RVV_15x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(15));
}

// gemm_RVV_15x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 15] @DRAM
// )
void gemm_RVV_15x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(15));
}

// gemm_RVV_15x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 15] @DRAM
// )
void gemm_RVV_15x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(15));
}

// gemm_RVV_15x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 15] @DRAM
// )
void gemm_RVV_15x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(15));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(15));
}

// gemm_RVV_16x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 16] @DRAM
// )
void gemm_RVV_16x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(16));
}

// gemm_RVV_16x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 16] @DRAM
// )
void gemm_RVV_16x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(16));
}

// gemm_RVV_16x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 16] @DRAM
// )
void gemm_RVV_16x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(16));
}

// gemm_RVV_16x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 16] @DRAM
// )
void gemm_RVV_16x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(16));
}

// gemm_RVV_17x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 17] @DRAM
// )
void gemm_RVV_17x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(1));
}

// gemm_RVV_17x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 17] @DRAM
// )
void gemm_RVV_17x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(1));
}

// gemm_RVV_17x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 17] @DRAM
// )
void gemm_RVV_17x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
}

// gemm_RVV_17x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 17] @DRAM
// )
void gemm_RVV_17x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
}

// gemm_RVV_18x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 18] @DRAM
// )
void gemm_RVV_18x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(2));
}

// gemm_RVV_18x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 18] @DRAM
// )
void gemm_RVV_18x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(2));
}

// gemm_RVV_18x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 18] @DRAM
// )
void gemm_RVV_18x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
}

// gemm_RVV_18x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 18] @DRAM
// )
void gemm_RVV_18x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
}

// gemm_RVV_19x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 19] @DRAM
// )
void gemm_RVV_19x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(3));
}

// gemm_RVV_19x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 19] @DRAM
// )
void gemm_RVV_19x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(3));
}

// gemm_RVV_19x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 19] @DRAM
// )
void gemm_RVV_19x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
}

// gemm_RVV_19x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 19] @DRAM
// )
void gemm_RVV_19x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_20x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 20] @DRAM
// )
void gemm_RVV_20x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(4));
}

// gemm_RVV_20x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 20] @DRAM
// )
void gemm_RVV_20x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(4));
}

// gemm_RVV_20x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 20] @DRAM
// )
void gemm_RVV_20x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(4));
}

// gemm_RVV_20x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 20] @DRAM
// )
void gemm_RVV_20x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(4));
}

// gemm_RVV_21x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 21] @DRAM
// )
void gemm_RVV_21x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(5));
}

// gemm_RVV_21x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 21] @DRAM
// )
void gemm_RVV_21x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(5));
}

// gemm_RVV_21x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 21] @DRAM
// )
void gemm_RVV_21x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(5));
}

// gemm_RVV_21x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 21] @DRAM
// )
void gemm_RVV_21x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(5));
}

// gemm_RVV_22x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 22] @DRAM
// )
void gemm_RVV_22x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(6));
}

// gemm_RVV_22x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 22] @DRAM
// )
void gemm_RVV_22x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(6));
}

// gemm_RVV_22x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 22] @DRAM
// )
void gemm_RVV_22x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(6));
}

// gemm_RVV_22x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 22] @DRAM
// )
void gemm_RVV_22x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(6));
}

// gemm_RVV_23x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 23] @DRAM
// )
void gemm_RVV_23x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(7));
}

// gemm_RVV_23x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 23] @DRAM
// )
void gemm_RVV_23x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(7));
}

// gemm_RVV_23x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 23] @DRAM
// )
void gemm_RVV_23x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(7));
}

// gemm_RVV_23x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 23] @DRAM
// )
void gemm_RVV_23x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(7));
}

// gemm_RVV_24x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 24] @DRAM
// )
void gemm_RVV_24x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(8));
}

// gemm_RVV_24x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 24] @DRAM
// )
void gemm_RVV_24x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(8));
}

// gemm_RVV_24x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 24] @DRAM
// )
void gemm_RVV_24x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(8));
}

// gemm_RVV_24x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 24] @DRAM
// )
void gemm_RVV_24x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(8));
}

// gemm_RVV_25x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 25] @DRAM
// )
void gemm_RVV_25x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(9));
}

// gemm_RVV_25x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 25] @DRAM
// )
void gemm_RVV_25x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(9));
}

// gemm_RVV_25x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 25] @DRAM
// )
void gemm_RVV_25x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(9));
}

// gemm_RVV_25x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 25] @DRAM
// )
void gemm_RVV_25x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(9));
}

// gemm_RVV_26x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 26] @DRAM
// )
void gemm_RVV_26x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(10));
}

// gemm_RVV_26x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 26] @DRAM
// )
void gemm_RVV_26x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(10));
}

// gemm_RVV_26x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 26] @DRAM
// )
void gemm_RVV_26x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(10));
}

// gemm_RVV_26x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 26] @DRAM
// )
void gemm_RVV_26x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(10));
}

// gemm_RVV_27x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 27] @DRAM
// )
void gemm_RVV_27x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(11));
}

// gemm_RVV_27x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 27] @DRAM
// )
void gemm_RVV_27x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(11));
}

// gemm_RVV_27x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 27] @DRAM
// )
void gemm_RVV_27x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(11));
}

// gemm_RVV_27x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 27] @DRAM
// )
void gemm_RVV_27x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(11));
}

// gemm_RVV_28x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 28] @DRAM
// )
void gemm_RVV_28x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(12));
}

// gemm_RVV_28x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 28] @DRAM
// )
void gemm_RVV_28x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(12));
}

// gemm_RVV_28x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 28] @DRAM
// )
void gemm_RVV_28x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(12));
}

// gemm_RVV_28x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 28] @DRAM
// )
void gemm_RVV_28x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(12));
}

// gemm_RVV_29x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 29] @DRAM
// )
void gemm_RVV_29x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(13));
}

// gemm_RVV_29x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 29] @DRAM
// )
void gemm_RVV_29x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(13));
}

// gemm_RVV_29x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 29] @DRAM
// )
void gemm_RVV_29x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(13));
}

// gemm_RVV_29x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 29] @DRAM
// )
void gemm_RVV_29x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(13));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_30x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 30] @DRAM
// )
void gemm_RVV_30x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(14));
}

// gemm_RVV_30x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 30] @DRAM
// )
void gemm_RVV_30x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(14));
}

// gemm_RVV_30x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 30] @DRAM
// )
void gemm_RVV_30x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(14));
}

// gemm_RVV_30x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 30] @DRAM
// )
void gemm_RVV_30x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(14));
}

// gemm_RVV_31x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 31] @DRAM
// )
void gemm_RVV_31x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(15));
}

// gemm_RVV_31x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 31] @DRAM
// )
void gemm_RVV_31x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(15));
}

// gemm_RVV_31x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 31] @DRAM
// )
void gemm_RVV_31x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(15));
}

// gemm_RVV_31x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 31] @DRAM
// )
void gemm_RVV_31x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(15));
}

// gemm_RVV_32x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 32] @DRAM
// )
void gemm_RVV_32x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
}

// gemm_RVV_32x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 32] @DRAM
// )
void gemm_RVV_32x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
}

// gemm_RVV_32x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 32] @DRAM
// )
void gemm_RVV_32x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
}

// gemm_RVV_32x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 32] @DRAM
// )
void gemm_RVV_32x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
}

// gemm_RVV_33x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 33] @DRAM
// )
void gemm_RVV_33x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(1));
}

// gemm_RVV_33x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 33] @DRAM
// )
void gemm_RVV_33x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(1));
}

// gemm_RVV_33x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 33] @DRAM
// )
void gemm_RVV_33x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(1));
}

// gemm_RVV_33x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 33] @DRAM
// )
void gemm_RVV_33x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(1));
}

// gemm_RVV_34x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 34] @DRAM
// )
void gemm_RVV_34x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(2));
}

// gemm_RVV_34x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 34] @DRAM
// )
void gemm_RVV_34x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(2));
}

// gemm_RVV_34x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 34] @DRAM
// )
void gemm_RVV_34x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(2));
}

// gemm_RVV_34x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 34] @DRAM
// )
void gemm_RVV_34x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(2));
}

// gemm_RVV_35x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 35] @DRAM
// )
void gemm_RVV_35x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(3));
}

// gemm_RVV_35x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 35] @DRAM
// )
void gemm_RVV_35x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(3));
}

// gemm_RVV_35x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 35] @DRAM
// )
void gemm_RVV_35x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(3));
}

// gemm_RVV_35x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 35] @DRAM
// )
void gemm_RVV_35x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(3));
}

// gemm_RVV_36x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 36] @DRAM
// )
void gemm_RVV_36x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(4));
}

// gemm_RVV_36x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 36] @DRAM
// )
void gemm_RVV_36x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(4));
}

// gemm_RVV_36x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 36] @DRAM
// )
void gemm_RVV_36x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(4));
}

// gemm_RVV_36x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 36] @DRAM
// )
void gemm_RVV_36x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(4));
}

// gemm_RVV_37x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 37] @DRAM
// )
void gemm_RVV_37x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(5));
}

// gemm_RVV_37x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 37] @DRAM
// )
void gemm_RVV_37x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(5));
}

// gemm_RVV_37x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 37] @DRAM
// )
void gemm_RVV_37x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(5));
}

// gemm_RVV_37x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 37] @DRAM
// )
void gemm_RVV_37x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(5));
}

// gemm_RVV_38x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 38] @DRAM
// )
void gemm_RVV_38x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(6));
}

// gemm_RVV_38x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 38] @DRAM
// )
void gemm_RVV_38x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(6));
}

// gemm_RVV_38x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 38] @DRAM
// )
void gemm_RVV_38x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(6));
}

// gemm_RVV_38x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 38] @DRAM
// )
void gemm_RVV_38x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(6));
}

// gemm_RVV_39x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 39] @DRAM
// )
void gemm_RVV_39x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(7));
}

// gemm_RVV_39x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 39] @DRAM
// )
void gemm_RVV_39x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(7));
}

// gemm_RVV_39x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 39] @DRAM
// )
void gemm_RVV_39x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(7));
}

// gemm_RVV_39x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 39] @DRAM
// )
void gemm_RVV_39x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(7));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_40x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 40] @DRAM
// )
void gemm_RVV_40x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(8));
}

// gemm_RVV_40x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 40] @DRAM
// )
void gemm_RVV_40x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(8));
}

// gemm_RVV_40x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 40] @DRAM
// )
void gemm_RVV_40x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(8));
}

// gemm_RVV_40x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 40] @DRAM
// )
void gemm_RVV_40x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(8));
}

// gemm_RVV_41x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 41] @DRAM
// )
void gemm_RVV_41x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(9));
}

// gemm_RVV_41x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 41] @DRAM
// )
void gemm_RVV_41x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(9));
}

// gemm_RVV_41x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 41] @DRAM
// )
void gemm_RVV_41x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(9));
}

// gemm_RVV_41x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 41] @DRAM
// )
void gemm_RVV_41x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(9));
}

// gemm_RVV_42x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 42] @DRAM
// )
void gemm_RVV_42x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(10));
}

// gemm_RVV_42x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 42] @DRAM
// )
void gemm_RVV_42x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(10));
}

// gemm_RVV_42x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 42] @DRAM
// )
void gemm_RVV_42x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(10));
}

// gemm_RVV_42x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 42] @DRAM
// )
void gemm_RVV_42x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(10));
}

// gemm_RVV_43x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 43] @DRAM
// )
void gemm_RVV_43x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(11));
}

// gemm_RVV_43x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 43] @DRAM
// )
void gemm_RVV_43x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(11));
}

// gemm_RVV_43x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 43] @DRAM
// )
void gemm_RVV_43x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(11));
}

// gemm_RVV_43x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 43] @DRAM
// )
void gemm_RVV_43x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(11));
}

// gemm_RVV_44x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 44] @DRAM
// )
void gemm_RVV_44x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(12));
}

// gemm_RVV_44x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 44] @DRAM
// )
void gemm_RVV_44x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(12));
}

// gemm_RVV_44x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 44] @DRAM
// )
void gemm_RVV_44x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(12));
}

// gemm_RVV_44x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 44] @DRAM
// )
void gemm_RVV_44x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(12));
}

// gemm_RVV_45x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 45] @DRAM
// )
void gemm_RVV_45x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(13));
}

// gemm_RVV_45x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 45] @DRAM
// )
void gemm_RVV_45x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(13));
}

// gemm_RVV_45x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 45] @DRAM
// )
void gemm_RVV_45x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(13));
}

// gemm_RVV_45x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 45] @DRAM
// )
void gemm_RVV_45x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(13));
}

// gemm_RVV_46x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 46] @DRAM
// )
void gemm_RVV_46x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(14));
}

// gemm_RVV_46x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 46] @DRAM
// )
void gemm_RVV_46x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(14));
}

// gemm_RVV_46x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 46] @DRAM
// )
void gemm_RVV_46x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(14));
}

// gemm_RVV_46x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 46] @DRAM
// )
void gemm_RVV_46x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(14));
}

// gemm_RVV_47x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 47] @DRAM
// )
void gemm_RVV_47x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(15));
}

// gemm_RVV_47x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 47] @DRAM
// )
void gemm_RVV_47x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(15));
}

// gemm_RVV_47x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 47] @DRAM
// )
void gemm_RVV_47x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(15));
}

// gemm_RVV_47x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 47] @DRAM
// )
void gemm_RVV_47x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(15));
}

// gemm_RVV_48x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 48] @DRAM
// )
void gemm_RVV_48x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
}

// gemm_RVV_48x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 48] @DRAM
// )
void gemm_RVV_48x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
}

// gemm_RVV_48x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 48] @DRAM
// )
void gemm_RVV_48x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
}

// gemm_RVV_48x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 48] @DRAM
// )
void gemm_RVV_48x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
}

// gemm_RVV_49x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 49] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 49] @DRAM
// )
void gemm_RVV_49x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(1));
}

// gemm_RVV_49x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 49] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 49] @DRAM
// )
void gemm_RVV_49x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(1));
}

// gemm_RVV_49x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 49] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 49] @DRAM
// )
void gemm_RVV_49x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(1));
}

// gemm_RVV_49x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 49] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 49] @DRAM
// )
void gemm_RVV_49x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(1));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_50x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 50] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 50] @DRAM
// )
void gemm_RVV_50x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(2));
}

// gemm_RVV_50x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 50] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 50] @DRAM
// )
void gemm_RVV_50x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(2));
}

// gemm_RVV_50x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 50] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 50] @DRAM
// )
void gemm_RVV_50x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(2));
}

// gemm_RVV_50x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 50] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 50] @DRAM
// )
void gemm_RVV_50x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(2));
}

// gemm_RVV_51x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 51] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 51] @DRAM
// )
void gemm_RVV_51x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(3));
}

// gemm_RVV_51x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 51] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 51] @DRAM
// )
void gemm_RVV_51x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(3));
}

// gemm_RVV_51x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 51] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 51] @DRAM
// )
void gemm_RVV_51x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(3));
}

// gemm_RVV_51x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 51] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 51] @DRAM
// )
void gemm_RVV_51x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(3));
}

// gemm_RVV_52x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 52] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 52] @DRAM
// )
void gemm_RVV_52x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(4));
}

// gemm_RVV_52x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 52] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 52] @DRAM
// )
void gemm_RVV_52x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(4));
}

// gemm_RVV_52x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 52] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 52] @DRAM
// )
void gemm_RVV_52x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(4));
}

// gemm_RVV_52x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 52] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 52] @DRAM
// )
void gemm_RVV_52x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(4));
}

// gemm_RVV_53x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 53] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 53] @DRAM
// )
void gemm_RVV_53x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(5));
}

// gemm_RVV_53x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 53] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 53] @DRAM
// )
void gemm_RVV_53x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(5));
}

// gemm_RVV_53x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 53] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 53] @DRAM
// )
void gemm_RVV_53x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(5));
}

// gemm_RVV_53x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 53] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 53] @DRAM
// )
void gemm_RVV_53x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(5));
}

// gemm_RVV_54x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 54] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 54] @DRAM
// )
void gemm_RVV_54x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(6));
}

// gemm_RVV_54x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 54] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 54] @DRAM
// )
void gemm_RVV_54x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(6));
}

// gemm_RVV_54x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 54] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 54] @DRAM
// )
void gemm_RVV_54x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(6));
}

// gemm_RVV_54x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 54] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 54] @DRAM
// )
void gemm_RVV_54x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(6));
}

// gemm_RVV_55x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 55] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 55] @DRAM
// )
void gemm_RVV_55x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(7));
}

// gemm_RVV_55x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 55] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 55] @DRAM
// )
void gemm_RVV_55x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(7));
}

// gemm_RVV_55x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 55] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 55] @DRAM
// )
void gemm_RVV_55x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(7));
}

// gemm_RVV_55x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 55] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 55] @DRAM
// )
void gemm_RVV_55x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(7));
}

// gemm_RVV_56x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 56] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 56] @DRAM
// )
void gemm_RVV_56x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(8));
}

// gemm_RVV_56x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 56] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 56] @DRAM
// )
void gemm_RVV_56x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(8));
}

// gemm_RVV_56x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 56] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 56] @DRAM
// )
void gemm_RVV_56x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(8));
}

// gemm_RVV_56x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 56] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 56] @DRAM
// )
void gemm_RVV_56x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(8));
}

// gemm_RVV_57x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 57] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 57] @DRAM
// )
void gemm_RVV_57x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(9));
}

// gemm_RVV_57x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 57] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 57] @DRAM
// )
void gemm_RVV_57x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(9));
}

// gemm_RVV_57x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 57] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 57] @DRAM
// )
void gemm_RVV_57x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(9));
}

// gemm_RVV_57x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 57] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 57] @DRAM
// )
void gemm_RVV_57x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(9));
}

// gemm_RVV_58x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 58] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 58] @DRAM
// )
void gemm_RVV_58x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(10));
}

// gemm_RVV_58x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 58] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 58] @DRAM
// )
void gemm_RVV_58x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(10));
}

// gemm_RVV_58x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 58] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 58] @DRAM
// )
void gemm_RVV_58x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(10));
}

// gemm_RVV_58x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 58] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 58] @DRAM
// )
void gemm_RVV_58x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(10));
}

// gemm_RVV_59x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 59] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 59] @DRAM
// )
void gemm_RVV_59x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(11));
}

// gemm_RVV_59x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 59] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 59] @DRAM
// )
void gemm_RVV_59x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(11));
}

// gemm_RVV_59x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 59] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 59] @DRAM
// )
void gemm_RVV_59x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(11));
}

// gemm_RVV_59x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 59] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 59] @DRAM
// )
void gemm_RVV_59x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(11));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
}

// gemm_RVV_60x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 60] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 60] @DRAM
// )
void gemm_RVV_60x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(12));
}

// gemm_RVV_60x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 60] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 60] @DRAM
// )
void gemm_RVV_60x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(12));
}

// gemm_RVV_60x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 60] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 60] @DRAM
// )
void gemm_RVV_60x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(12));
}

// gemm_RVV_60x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 60] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 60] @DRAM
// )
void gemm_RVV_60x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(12));
}

// gemm_RVV_61x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 61] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 61] @DRAM
// )
void gemm_RVV_61x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(13));
}

// gemm_RVV_61x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 61] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 61] @DRAM
// )
void gemm_RVV_61x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(13));
}

// gemm_RVV_61x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 61] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 61] @DRAM
// )
void gemm_RVV_61x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(13));
}

// gemm_RVV_61x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 61] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 61] @DRAM
// )
void gemm_RVV_61x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(13));
}

// gemm_RVV_62x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 62] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 62] @DRAM
// )
void gemm_RVV_62x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(14));
}

// gemm_RVV_62x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 62] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 62] @DRAM
// )
void gemm_RVV_62x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(14));
}

// gemm_RVV_62x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 62] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 62] @DRAM
// )
void gemm_RVV_62x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(14));
}

// gemm_RVV_62x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 62] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 62] @DRAM
// )
void gemm_RVV_62x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(14));
}

// gemm_RVV_63x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 63] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 63] @DRAM
// )
void gemm_RVV_63x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(15));
}

// gemm_RVV_63x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 63] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 63] @DRAM
// )
void gemm_RVV_63x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(15));
}

// gemm_RVV_63x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 63] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 63] @DRAM
// )
void gemm_RVV_63x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(15));
}

// gemm_RVV_63x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 63] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 63] @DRAM
// )
void gemm_RVV_63x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[48],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_regt_1,(15));
}

// gemm_RVV_64x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 64] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 64] @DRAM
// )
void gemm_RVV_64x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
}

// gemm_RVV_64x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 64] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 64] @DRAM
// )
void gemm_RVV_64x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
}

// gemm_RVV_64x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 64] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 64] @DRAM
// )
void gemm_RVV_64x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
}

// gemm_RVV_64x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 64] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 64] @DRAM
// )
void gemm_RVV_64x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
}

// gemm_RVV_65x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 65] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 65] @DRAM
// )
void gemm_RVV_65x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(1));
}

// gemm_RVV_65x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 65] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 65] @DRAM
// )
void gemm_RVV_65x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(1));
}

// gemm_RVV_65x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 65] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 65] @DRAM
// )
void gemm_RVV_65x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(1));
}

// gemm_RVV_65x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 65] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 65] @DRAM
// )
void gemm_RVV_65x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(1));
}

// gemm_RVV_66x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 66] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 66] @DRAM
// )
void gemm_RVV_66x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(2));
}

// gemm_RVV_66x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 66] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 66] @DRAM
// )
void gemm_RVV_66x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(2));
}

// gemm_RVV_66x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 66] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 66] @DRAM
// )
void gemm_RVV_66x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(2));
}

// gemm_RVV_66x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 66] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 66] @DRAM
// )
void gemm_RVV_66x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(2));
}

// gemm_RVV_67x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 67] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 67] @DRAM
// )
void gemm_RVV_67x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(3));
}

// gemm_RVV_67x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 67] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 67] @DRAM
// )
void gemm_RVV_67x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(3));
}

// gemm_RVV_67x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 67] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 67] @DRAM
// )
void gemm_RVV_67x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(3));
}

// gemm_RVV_67x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 67] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 67] @DRAM
// )
void gemm_RVV_67x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(3));
}

// gemm_RVV_68x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 68] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 68] @DRAM
// )
void gemm_RVV_68x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(4));
}

// gemm_RVV_68x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 68] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 68] @DRAM
// )
void gemm_RVV_68x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(4));
}

// gemm_RVV_68x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 68] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 68] @DRAM
// )
void gemm_RVV_68x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(4));
}

// gemm_RVV_68x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 68] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 68] @DRAM
// )
void gemm_RVV_68x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(4));
}

// gemm_RVV_69x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 69] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 69] @DRAM
// )
void gemm_RVV_69x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(5));
}

// gemm_RVV_69x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 69] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 69] @DRAM
// )
void gemm_RVV_69x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(5));
}

// gemm_RVV_69x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 69] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 69] @DRAM
// )
void gemm_RVV_69x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(5));
}

// gemm_RVV_69x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 69] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 69] @DRAM
// )
void gemm_RVV_69x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(5));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
}

// gemm_RVV_70x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 70] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 70] @DRAM
// )
void gemm_RVV_70x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(6));
}

// gemm_RVV_70x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 70] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 70] @DRAM
// )
void gemm_RVV_70x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(6));
}

// gemm_RVV_70x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 70] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 70] @DRAM
// )
void gemm_RVV_70x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(6));
}

// gemm_RVV_70x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 70] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 70] @DRAM
// )
void gemm_RVV_70x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(6));
}

// gemm_RVV_71x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 71] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 71] @DRAM
// )
void gemm_RVV_71x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(7));
}

// gemm_RVV_71x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 71] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 71] @DRAM
// )
void gemm_RVV_71x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(7));
}

// gemm_RVV_71x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 71] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 71] @DRAM
// )
void gemm_RVV_71x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(7));
}

// gemm_RVV_71x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 71] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 71] @DRAM
// )
void gemm_RVV_71x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(7));
}

// gemm_RVV_72x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 72] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 72] @DRAM
// )
void gemm_RVV_72x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(8));
}

// gemm_RVV_72x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 72] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 72] @DRAM
// )
void gemm_RVV_72x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(8));
}

// gemm_RVV_72x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 72] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 72] @DRAM
// )
void gemm_RVV_72x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(8));
}

// gemm_RVV_72x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 72] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 72] @DRAM
// )
void gemm_RVV_72x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(8));
}

// gemm_RVV_73x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 73] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 73] @DRAM
// )
void gemm_RVV_73x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(9));
}

// gemm_RVV_73x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 73] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 73] @DRAM
// )
void gemm_RVV_73x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(9));
}

// gemm_RVV_73x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 73] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 73] @DRAM
// )
void gemm_RVV_73x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(9));
}

// gemm_RVV_73x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 73] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 73] @DRAM
// )
void gemm_RVV_73x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(9));
}

// gemm_RVV_74x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 74] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 74] @DRAM
// )
void gemm_RVV_74x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(10));
}

// gemm_RVV_74x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 74] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 74] @DRAM
// )
void gemm_RVV_74x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(10));
}

// gemm_RVV_74x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 74] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 74] @DRAM
// )
void gemm_RVV_74x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(10));
}

// gemm_RVV_74x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 74] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 74] @DRAM
// )
void gemm_RVV_74x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(10));
}

// gemm_RVV_75x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 75] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 75] @DRAM
// )
void gemm_RVV_75x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(11));
}

// gemm_RVV_75x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 75] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 75] @DRAM
// )
void gemm_RVV_75x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(11));
}

// gemm_RVV_75x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 75] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 75] @DRAM
// )
void gemm_RVV_75x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(11));
}

// gemm_RVV_75x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 75] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 75] @DRAM
// )
void gemm_RVV_75x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(11));
}

// gemm_RVV_76x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 76] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 76] @DRAM
// )
void gemm_RVV_76x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(12));
}

// gemm_RVV_76x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 76] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 76] @DRAM
// )
void gemm_RVV_76x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(12));
}

// gemm_RVV_76x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 76] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 76] @DRAM
// )
void gemm_RVV_76x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(12));
}

// gemm_RVV_76x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 76] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 76] @DRAM
// )
void gemm_RVV_76x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(12));
}

// gemm_RVV_77x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 77] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 77] @DRAM
// )
void gemm_RVV_77x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(13));
}

// gemm_RVV_77x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 77] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 77] @DRAM
// )
void gemm_RVV_77x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(13));
}

// gemm_RVV_77x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 77] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 77] @DRAM
// )
void gemm_RVV_77x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(13));
}

// gemm_RVV_77x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 77] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 77] @DRAM
// )
void gemm_RVV_77x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(13));
}

// gemm_RVV_78x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 78] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 78] @DRAM
// )
void gemm_RVV_78x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(14));
}

// gemm_RVV_78x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 78] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 78] @DRAM
// )
void gemm_RVV_78x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(14));
}

// gemm_RVV_78x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 78] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 78] @DRAM
// )
void gemm_RVV_78x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(14));
}

// gemm_RVV_78x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 78] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 78] @DRAM
// )
void gemm_RVV_78x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(14));
}

// gemm_RVV_79x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 79] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 79] @DRAM
// )
void gemm_RVV_79x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(15));
}

// gemm_RVV_79x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 79] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 79] @DRAM
// )
void gemm_RVV_79x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(15));
}

// gemm_RVV_79x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 79] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 79] @DRAM
// )
void gemm_RVV_79x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(15));
}

// gemm_RVV_79x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 79] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 79] @DRAM
// )
void gemm_RVV_79x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[64],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_regt_1,(15));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
}

// gemm_RVV_80x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 80] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 80] @DRAM
// )
void gemm_RVV_80x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
}

// gemm_RVV_80x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 80] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 80] @DRAM
// )
void gemm_RVV_80x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
}

// gemm_RVV_80x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 80] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 80] @DRAM
// )
void gemm_RVV_80x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
}

// gemm_RVV_80x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 80] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 80] @DRAM
// )
void gemm_RVV_80x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
}

// gemm_RVV_81x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 81] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 81] @DRAM
// )
void gemm_RVV_81x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(1));
}

// gemm_RVV_81x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 81] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 81] @DRAM
// )
void gemm_RVV_81x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(1));
}

// gemm_RVV_81x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 81] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 81] @DRAM
// )
void gemm_RVV_81x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(1));
}

// gemm_RVV_81x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 81] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 81] @DRAM
// )
void gemm_RVV_81x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(1));
}

// gemm_RVV_82x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 82] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 82] @DRAM
// )
void gemm_RVV_82x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(2));
}

// gemm_RVV_82x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 82] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 82] @DRAM
// )
void gemm_RVV_82x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(2));
}

// gemm_RVV_82x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 82] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 82] @DRAM
// )
void gemm_RVV_82x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(2));
}

// gemm_RVV_82x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 82] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 82] @DRAM
// )
void gemm_RVV_82x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(2));
}

// gemm_RVV_83x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 83] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 83] @DRAM
// )
void gemm_RVV_83x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(3));
}

// gemm_RVV_83x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 83] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 83] @DRAM
// )
void gemm_RVV_83x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(3));
}

// gemm_RVV_83x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 83] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 83] @DRAM
// )
void gemm_RVV_83x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(3));
}

// gemm_RVV_83x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 83] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 83] @DRAM
// )
void gemm_RVV_83x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(3));
}

// gemm_RVV_84x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 84] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 84] @DRAM
// )
void gemm_RVV_84x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(4));
}

// gemm_RVV_84x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 84] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 84] @DRAM
// )
void gemm_RVV_84x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(4));
}

// gemm_RVV_84x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 84] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 84] @DRAM
// )
void gemm_RVV_84x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(4));
}

// gemm_RVV_84x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 84] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 84] @DRAM
// )
void gemm_RVV_84x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(4));
}

// gemm_RVV_85x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 85] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 85] @DRAM
// )
void gemm_RVV_85x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(5));
}

// gemm_RVV_85x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 85] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 85] @DRAM
// )
void gemm_RVV_85x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(5));
}

// gemm_RVV_85x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 85] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 85] @DRAM
// )
void gemm_RVV_85x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(5));
}

// gemm_RVV_85x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 85] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 85] @DRAM
// )
void gemm_RVV_85x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(5));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(5));
}

// gemm_RVV_86x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 86] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 86] @DRAM
// )
void gemm_RVV_86x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(6));
}

// gemm_RVV_86x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 86] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 86] @DRAM
// )
void gemm_RVV_86x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(6));
}

// gemm_RVV_86x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 86] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 86] @DRAM
// )
void gemm_RVV_86x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(6));
}

// gemm_RVV_86x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 86] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 86] @DRAM
// )
void gemm_RVV_86x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(6));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(6));
}

// gemm_RVV_87x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 87] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 87] @DRAM
// )
void gemm_RVV_87x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(7));
}

// gemm_RVV_87x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 87] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 87] @DRAM
// )
void gemm_RVV_87x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(7));
}

// gemm_RVV_87x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 87] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 87] @DRAM
// )
void gemm_RVV_87x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(7));
}

// gemm_RVV_87x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 87] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 87] @DRAM
// )
void gemm_RVV_87x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(7));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(7));
}

// gemm_RVV_88x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 88] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 88] @DRAM
// )
void gemm_RVV_88x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(8));
}

// gemm_RVV_88x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 88] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 88] @DRAM
// )
void gemm_RVV_88x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(8));
}

// gemm_RVV_88x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 88] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 88] @DRAM
// )
void gemm_RVV_88x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(8));
}

// gemm_RVV_88x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 88] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 88] @DRAM
// )
void gemm_RVV_88x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(8));
}

// gemm_RVV_89x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 89] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 89] @DRAM
// )
void gemm_RVV_89x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(9));
}

// gemm_RVV_89x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 89] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 89] @DRAM
// )
void gemm_RVV_89x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(9));
}

// gemm_RVV_89x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 89] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 89] @DRAM
// )
void gemm_RVV_89x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(9));
}

// gemm_RVV_89x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 89] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 89] @DRAM
// )
void gemm_RVV_89x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(9));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(9));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
}

// gemm_RVV_90x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 90] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 90] @DRAM
// )
void gemm_RVV_90x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(10));
}

// gemm_RVV_90x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 90] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 90] @DRAM
// )
void gemm_RVV_90x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(10));
}

// gemm_RVV_90x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 90] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 90] @DRAM
// )
void gemm_RVV_90x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(10));
}

// gemm_RVV_90x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 90] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 90] @DRAM
// )
void gemm_RVV_90x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(10));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(10));
}

// gemm_RVV_91x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 91] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 91] @DRAM
// )
void gemm_RVV_91x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(11));
}

// gemm_RVV_91x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 91] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 91] @DRAM
// )
void gemm_RVV_91x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(11));
}

// gemm_RVV_91x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 91] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 91] @DRAM
// )
void gemm_RVV_91x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(11));
}

// gemm_RVV_91x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 91] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 91] @DRAM
// )
void gemm_RVV_91x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(11));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(11));
}

// gemm_RVV_92x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 92] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 92] @DRAM
// )
void gemm_RVV_92x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(12));
}

// gemm_RVV_92x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 92] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 92] @DRAM
// )
void gemm_RVV_92x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(12));
}

// gemm_RVV_92x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 92] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 92] @DRAM
// )
void gemm_RVV_92x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(12));
}

// gemm_RVV_92x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 92] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 92] @DRAM
// )
void gemm_RVV_92x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(12));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(12));
}

// gemm_RVV_93x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 93] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 93] @DRAM
// )
void gemm_RVV_93x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(13));
}

// gemm_RVV_93x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 93] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 93] @DRAM
// )
void gemm_RVV_93x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(13));
}

// gemm_RVV_93x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 93] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 93] @DRAM
// )
void gemm_RVV_93x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(13));
}

// gemm_RVV_93x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 93] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 93] @DRAM
// )
void gemm_RVV_93x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(13));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(13));
}

// gemm_RVV_94x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 94] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 94] @DRAM
// )
void gemm_RVV_94x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(14));
}

// gemm_RVV_94x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 94] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 94] @DRAM
// )
void gemm_RVV_94x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(14));
}

// gemm_RVV_94x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 94] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 94] @DRAM
// )
void gemm_RVV_94x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(14));
}

// gemm_RVV_94x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 94] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 94] @DRAM
// )
void gemm_RVV_94x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(14));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(14));
}

// gemm_RVV_95x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 95] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 95] @DRAM
// )
void gemm_RVV_95x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(15));
}

// gemm_RVV_95x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 95] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 95] @DRAM
// )
void gemm_RVV_95x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(15));
}

// gemm_RVV_95x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 95] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 95] @DRAM
// )
void gemm_RVV_95x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(15));
}

// gemm_RVV_95x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 95] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 95] @DRAM
// )
void gemm_RVV_95x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[80],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(15));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_regt_1,(15));
}

// gemm_RVV_96x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 96] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 96] @DRAM
// )
void gemm_RVV_96x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_reg_5;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_0_5;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_5 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_reg_5 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_0_5 = __riscv_vfmacc_vv_f16m1(C_reg_0_5, A_reg_5, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_reg_0_5,(16));
}

// gemm_RVV_96x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 96] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 96] @DRAM
// )
void gemm_RVV_96x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_reg_5;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_0_5;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_0_5 = __riscv_vle16_v_f16m1(&C.data[80],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_reg_5 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_0_5 = __riscv_vfmacc_vv_f16m1(C_reg_0_5, A_reg_5, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_reg_0_5,(16));
}

// gemm_RVV_96x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 96] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 96] @DRAM
// )
void gemm_RVV_96x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_reg_5;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_reg_5 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_0_5 = __riscv_vfmacc_vv_f16m1(C_reg_0_5, A_reg_5, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_reg_1_5 = __riscv_vfmacc_vv_f16m1(C_reg_1_5, A_reg_5, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_reg_0_5,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_reg_1_5,(16));
}

// gemm_RVV_96x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 96] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 96] @DRAM
// )
void gemm_RVV_96x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_reg_5;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
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
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[64],(16));
C_reg_0_5 = __riscv_vle16_v_f16m1(&C.data[80],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 64],(16));
C_reg_1_5 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 80],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(16));
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  A_reg_5 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 80],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(16));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(16));
  C_reg_0_5 = __riscv_vfmacc_vv_f16m1(C_reg_0_5, A_reg_5, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(16));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(16));
  C_reg_1_5 = __riscv_vfmacc_vv_f16m1(C_reg_1_5, A_reg_5, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C.data[80], C_reg_0_5,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 80], C_reg_1_5,(16));
}

// gemm_RVV_9x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 9] @DRAM
// )
void gemm_RVV_9x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(9));
}

// gemm_RVV_9x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 9] @DRAM
// )
void gemm_RVV_9x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(9));
}

// gemm_RVV_9x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 9] @DRAM
// )
void gemm_RVV_9x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(9));
}

// gemm_RVV_9x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 9] @DRAM
// )
void gemm_RVV_9x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(9));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(9));
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(9));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(9));
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
