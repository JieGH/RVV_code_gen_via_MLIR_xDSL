#ifndef GEMM_HEADERS_H
#define GEMM_HEADERS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <riscv_vector.h>

#define GET_PTR(ptr, off) (ptr + off)
#define GET_BETA(pc, beta) (pc == 0 ? beta : 1.0f)

extern "C" {
void gemm_microkernel_v_fp32_1x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x5_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x6_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x7_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x8_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x5_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x6_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x7_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x8_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x5_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x6_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x7_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x8_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x5_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x6_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x7_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x8_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x5_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x6_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x7_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x8_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x5_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x6_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x7_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x8_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x5_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x6_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x7_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x8_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x5_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x6_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x7_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x8_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x5_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x6_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x7_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_1x8_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x5_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x6_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x7_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_2x8_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x5_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x6_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x7_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_3x8_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x5_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x6_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x7_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_4x8_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x5_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x6_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x7_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_5x8_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x5_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x6_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x7_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_6x8_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x5_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x6_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x7_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_7x8_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x5_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x6_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x7_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_8x8_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);

void pack_CB_v_if_fp32_MR8_NR8(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR8_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR8_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR8_NR8(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR8_NR8(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR8_NR8(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR8_NR7(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR8_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR8_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR8_NR7(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR8_NR7(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR8_NR7(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR8_NR6(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR8_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR8_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR8_NR6(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR8_NR6(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR8_NR6(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR8_NR5(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR8_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR8_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR8_NR5(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR8_NR5(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR8_NR5(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR8_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR8_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR8_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR8_NR4(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR8_NR4(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR8_NR4(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR8_NR3(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR8_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR8_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR8_NR3(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR8_NR3(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR8_NR3(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR8_NR2(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR8_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR8_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR8_NR2(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR8_NR2(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR8_NR2(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR7_NR8(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR7_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR7_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR7_NR8(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR7_NR8(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR7_NR8(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR7_NR7(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR7_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR7_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR7_NR7(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR7_NR7(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR7_NR7(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR7_NR6(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR7_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR7_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR7_NR6(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR7_NR6(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR7_NR6(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR7_NR5(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR7_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR7_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR7_NR5(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR7_NR5(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR7_NR5(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR7_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR7_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR7_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR7_NR4(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR7_NR4(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR7_NR4(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR7_NR3(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR7_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR7_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR7_NR3(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR7_NR3(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR7_NR3(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR7_NR2(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR7_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR7_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR7_NR2(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR7_NR2(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR7_NR2(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR6_NR8(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR6_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR6_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR6_NR8(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR6_NR8(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR6_NR8(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR6_NR7(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR6_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR6_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR6_NR7(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR6_NR7(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR6_NR7(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR6_NR6(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR6_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR6_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR6_NR6(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR6_NR6(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR6_NR6(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR6_NR5(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR6_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR6_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR6_NR5(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR6_NR5(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR6_NR5(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR6_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR6_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR6_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR6_NR4(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR6_NR4(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR6_NR4(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR6_NR3(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR6_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR6_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR6_NR3(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR6_NR3(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR6_NR3(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR6_NR2(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR6_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR6_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR6_NR2(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR6_NR2(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR6_NR2(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR5_NR8(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR5_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR5_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR5_NR8(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR5_NR8(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR5_NR8(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR5_NR7(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR5_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR5_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR5_NR7(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR5_NR7(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR5_NR7(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR5_NR6(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR5_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR5_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR5_NR6(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR5_NR6(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR5_NR6(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR5_NR5(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR5_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR5_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR5_NR5(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR5_NR5(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR5_NR5(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR5_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR5_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR5_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR5_NR4(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR5_NR4(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR5_NR4(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR5_NR3(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR5_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR5_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR5_NR3(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR5_NR3(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR5_NR3(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR5_NR2(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR5_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR5_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR5_NR2(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR5_NR2(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR5_NR2(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR4_NR8(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR4_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR4_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR4_NR8(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR4_NR8(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR4_NR8(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR4_NR7(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR4_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR4_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR4_NR7(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR4_NR7(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR4_NR7(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR4_NR6(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR4_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR4_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR4_NR6(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR4_NR6(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR4_NR6(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR4_NR5(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR4_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR4_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR4_NR5(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR4_NR5(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR4_NR5(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR4_NR4(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR4_NR4(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR4_NR3(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR4_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR4_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR4_NR3(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR4_NR3(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR4_NR3(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR4_NR2(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR4_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR4_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR4_NR2(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR4_NR2(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR4_NR2(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR3_NR8(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR3_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR3_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR3_NR8(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR3_NR8(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR3_NR8(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR3_NR7(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR3_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR3_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR3_NR7(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR3_NR7(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR3_NR7(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR3_NR6(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR3_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR3_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR3_NR6(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR3_NR6(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR3_NR6(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR3_NR5(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR3_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR3_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR3_NR5(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR3_NR5(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR3_NR5(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR3_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR3_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR3_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR3_NR4(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR3_NR4(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR3_NR4(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR3_NR3(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR3_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR3_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR3_NR3(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR3_NR3(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR3_NR3(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR3_NR2(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR3_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR3_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR3_NR2(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR3_NR2(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR3_NR2(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR2_NR8(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR2_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR2_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR2_NR8(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR2_NR8(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR2_NR8(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR2_NR7(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR2_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR2_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR2_NR7(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR2_NR7(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR2_NR7(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR2_NR6(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR2_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR2_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR2_NR6(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR2_NR6(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR2_NR6(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR2_NR5(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR2_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR2_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR2_NR5(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR2_NR5(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR2_NR5(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR2_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR2_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR2_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR2_NR4(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR2_NR4(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR2_NR4(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR2_NR3(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR2_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR2_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR2_NR3(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR2_NR3(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR2_NR3(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

void pack_CB_v_if_fp32_MR2_NR2(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR2_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR2_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR2_NR2(float*, float*, size_t, size_t, size_t);

void gemm_microkernel_v_dispatch_fp32_MR2_NR2(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);
void gemm_microkernel_v_wrapper_fp32_MR2_NR2(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);

}

#endif // GEMM_HEADERS_H
