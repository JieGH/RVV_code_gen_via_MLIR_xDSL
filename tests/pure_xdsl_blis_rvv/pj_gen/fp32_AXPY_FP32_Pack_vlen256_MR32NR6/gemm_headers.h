#ifndef GEMM_HEADERS_H
#define GEMM_HEADERS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <riscv_vector.h>
#include <chrono>
#include "perf_counters.h"

#define GET_PTR(ptr, off) (ptr + off)
#define GET_BETA(pc, beta) (pc == 0 ? beta : (float)1)

extern "C" {
void gemm_microkernel_v_fp32_32x6_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_32x6_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR1(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR1(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR1(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR1(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR2(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR2(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR3(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR3(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR4(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR5(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR5(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR6(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR6(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR7(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR7(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR8(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR8(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR9(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR9(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR9(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR9(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR10(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR10(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR10(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR10(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR11(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR11(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR11(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR11(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR12(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR12(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR12(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR12(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR13(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR13(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR13(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR13(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR14(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR14(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR14(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR14(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR15(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR15(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR15(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR15(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR16(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR16(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR16(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR16(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR17(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR17(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR17(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR17(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR18(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR18(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR18(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR18(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR19(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR19(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR19(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR19(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR20(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR20(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR20(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR20(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR21(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR21(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR21(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR21(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR22(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR22(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR22(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR22(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR23(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR23(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR23(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR23(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR24(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR24(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR24(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR24(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR25(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR25(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR25(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR25(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR26(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR26(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR26(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR26(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR27(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR27(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR27(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR27(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR28(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR28(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR28(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR28(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR29(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR29(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR29(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR29(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR30(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR30(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR30(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR30(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR31(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR31(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR31(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR31(float*, float*, size_t, size_t, size_t);

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR32(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR32(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR32(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR32(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_generic(size_t cur_mr, size_t cur_nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc, size_t max_mr, size_t max_nr);

}

#endif // GEMM_HEADERS_H
