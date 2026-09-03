#include "kernels_RVV_80x2_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


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
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B.data[(k) * (B.strides[0])], A_reg_1,(16));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B.data[(k) * (B.strides[0])], A_reg_2,(16));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B.data[(k) * (B.strides[0])], A_reg_3,(16));
  C_reg_0_4 = __riscv_vfmacc_vf_f16m1(C_reg_0_4, B.data[(k) * (B.strides[0])], A_reg_4,(16));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(16));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B.data[(k) * (B.strides[0]) + 1], A_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B.data[(k) * (B.strides[0]) + 1], A_reg_2,(16));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B.data[(k) * (B.strides[0]) + 1], A_reg_3,(16));
  C_reg_1_4 = __riscv_vfmacc_vf_f16m1(C_reg_1_4, B.data[(k) * (B.strides[0]) + 1], A_reg_4,(16));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 64],(16));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B.data[(k) * (B.strides[0])], A_reg_1,(16));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B.data[(k) * (B.strides[0])], A_reg_2,(16));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B.data[(k) * (B.strides[0])], A_reg_3,(16));
  C_reg_0_4 = __riscv_vfmacc_vf_f16m1(C_reg_0_4, B.data[(k) * (B.strides[0])], A_reg_4,(16));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(16));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B.data[(k) * (B.strides[0]) + 1], A_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B.data[(k) * (B.strides[0]) + 1], A_reg_2,(16));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B.data[(k) * (B.strides[0]) + 1], A_reg_3,(16));
  C_reg_1_4 = __riscv_vfmacc_vf_f16m1(C_reg_1_4, B.data[(k) * (B.strides[0]) + 1], A_reg_4,(16));
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
