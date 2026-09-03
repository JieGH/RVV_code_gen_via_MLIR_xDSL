#include "kernels_RVV_64x2_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_64x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 64] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 64] @DRAM
// )
void gemm_RVV_64x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (64)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (64) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (64) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (64) + 48],(16));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (2)], A_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (2)], A_reg_1,(16));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (2)], A_reg_2,(16));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (2)], A_reg_3,(16));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (2) + 1], A_reg_0,(16));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (2) + 1], A_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (2) + 1], A_reg_2,(16));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B[(k) * (2) + 1], A_reg_3,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C[ldc + 48], C_reg_1_3,(16));
}

// gemm_RVV_64x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 64] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 64] @DRAM
// )
void gemm_RVV_64x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C[ldc + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (64)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (64) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (64) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (64) + 48],(16));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (2)], A_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (2)], A_reg_1,(16));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (2)], A_reg_2,(16));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (2)], A_reg_3,(16));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (2) + 1], A_reg_0,(16));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (2) + 1], A_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (2) + 1], A_reg_2,(16));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B[(k) * (2) + 1], A_reg_3,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C[ldc + 48], C_reg_1_3,(16));
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
