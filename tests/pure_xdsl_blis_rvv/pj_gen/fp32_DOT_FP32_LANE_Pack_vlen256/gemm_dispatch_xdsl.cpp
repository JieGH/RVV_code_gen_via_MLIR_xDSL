#include "gemm_headers.h"

extern "C" {

void gemm_microkernel_v_dispatch_generic(size_t cur_mr, size_t cur_nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc, size_t max_mr, size_t max_nr) {
    if (beta == (float)0) {
        if (cur_mr == 1 && cur_nr == 1) { gemm_microkernel_v_fp32_1x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 2) { gemm_microkernel_v_fp32_1x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 3) { gemm_microkernel_v_fp32_1x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 4) { gemm_microkernel_v_fp32_1x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 5) { gemm_microkernel_v_fp32_1x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 6) { gemm_microkernel_v_fp32_1x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 7) { gemm_microkernel_v_fp32_1x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 8) { gemm_microkernel_v_fp32_1x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 9) { gemm_microkernel_v_fp32_1x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 10) { gemm_microkernel_v_fp32_1x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 11) { gemm_microkernel_v_fp32_1x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 12) { gemm_microkernel_v_fp32_1x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 13) { gemm_microkernel_v_fp32_1x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 14) { gemm_microkernel_v_fp32_1x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 15) { gemm_microkernel_v_fp32_1x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 16) { gemm_microkernel_v_fp32_1x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 1) { gemm_microkernel_v_fp32_2x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 2) { gemm_microkernel_v_fp32_2x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 3) { gemm_microkernel_v_fp32_2x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 4) { gemm_microkernel_v_fp32_2x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 5) { gemm_microkernel_v_fp32_2x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 6) { gemm_microkernel_v_fp32_2x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 7) { gemm_microkernel_v_fp32_2x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 8) { gemm_microkernel_v_fp32_2x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 9) { gemm_microkernel_v_fp32_2x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 10) { gemm_microkernel_v_fp32_2x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 11) { gemm_microkernel_v_fp32_2x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 12) { gemm_microkernel_v_fp32_2x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 13) { gemm_microkernel_v_fp32_2x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 14) { gemm_microkernel_v_fp32_2x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 15) { gemm_microkernel_v_fp32_2x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 16) { gemm_microkernel_v_fp32_2x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 1) { gemm_microkernel_v_fp32_3x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 2) { gemm_microkernel_v_fp32_3x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 3) { gemm_microkernel_v_fp32_3x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 4) { gemm_microkernel_v_fp32_3x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 5) { gemm_microkernel_v_fp32_3x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 6) { gemm_microkernel_v_fp32_3x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 7) { gemm_microkernel_v_fp32_3x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 8) { gemm_microkernel_v_fp32_3x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 9) { gemm_microkernel_v_fp32_3x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 10) { gemm_microkernel_v_fp32_3x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 11) { gemm_microkernel_v_fp32_3x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 12) { gemm_microkernel_v_fp32_3x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 13) { gemm_microkernel_v_fp32_3x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 14) { gemm_microkernel_v_fp32_3x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 15) { gemm_microkernel_v_fp32_3x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 16) { gemm_microkernel_v_fp32_3x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 1) { gemm_microkernel_v_fp32_4x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 2) { gemm_microkernel_v_fp32_4x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 3) { gemm_microkernel_v_fp32_4x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 4) { gemm_microkernel_v_fp32_4x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 5) { gemm_microkernel_v_fp32_4x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 6) { gemm_microkernel_v_fp32_4x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 7) { gemm_microkernel_v_fp32_4x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 8) { gemm_microkernel_v_fp32_4x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 9) { gemm_microkernel_v_fp32_4x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 10) { gemm_microkernel_v_fp32_4x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 11) { gemm_microkernel_v_fp32_4x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 12) { gemm_microkernel_v_fp32_4x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 13) { gemm_microkernel_v_fp32_4x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 14) { gemm_microkernel_v_fp32_4x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 15) { gemm_microkernel_v_fp32_4x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 16) { gemm_microkernel_v_fp32_4x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 1) { gemm_microkernel_v_fp32_5x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 2) { gemm_microkernel_v_fp32_5x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 3) { gemm_microkernel_v_fp32_5x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 4) { gemm_microkernel_v_fp32_5x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 5) { gemm_microkernel_v_fp32_5x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 6) { gemm_microkernel_v_fp32_5x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 7) { gemm_microkernel_v_fp32_5x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 8) { gemm_microkernel_v_fp32_5x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 9) { gemm_microkernel_v_fp32_5x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 10) { gemm_microkernel_v_fp32_5x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 11) { gemm_microkernel_v_fp32_5x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 12) { gemm_microkernel_v_fp32_5x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 13) { gemm_microkernel_v_fp32_5x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 14) { gemm_microkernel_v_fp32_5x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 15) { gemm_microkernel_v_fp32_5x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 16) { gemm_microkernel_v_fp32_5x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 1) { gemm_microkernel_v_fp32_6x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 2) { gemm_microkernel_v_fp32_6x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 3) { gemm_microkernel_v_fp32_6x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 4) { gemm_microkernel_v_fp32_6x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 5) { gemm_microkernel_v_fp32_6x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 6) { gemm_microkernel_v_fp32_6x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 7) { gemm_microkernel_v_fp32_6x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 8) { gemm_microkernel_v_fp32_6x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 9) { gemm_microkernel_v_fp32_6x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 10) { gemm_microkernel_v_fp32_6x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 11) { gemm_microkernel_v_fp32_6x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 12) { gemm_microkernel_v_fp32_6x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 13) { gemm_microkernel_v_fp32_6x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 14) { gemm_microkernel_v_fp32_6x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 15) { gemm_microkernel_v_fp32_6x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 16) { gemm_microkernel_v_fp32_6x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 1) { gemm_microkernel_v_fp32_7x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 2) { gemm_microkernel_v_fp32_7x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 3) { gemm_microkernel_v_fp32_7x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 4) { gemm_microkernel_v_fp32_7x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 5) { gemm_microkernel_v_fp32_7x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 6) { gemm_microkernel_v_fp32_7x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 7) { gemm_microkernel_v_fp32_7x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 8) { gemm_microkernel_v_fp32_7x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 9) { gemm_microkernel_v_fp32_7x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 10) { gemm_microkernel_v_fp32_7x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 11) { gemm_microkernel_v_fp32_7x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 12) { gemm_microkernel_v_fp32_7x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 13) { gemm_microkernel_v_fp32_7x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 14) { gemm_microkernel_v_fp32_7x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 15) { gemm_microkernel_v_fp32_7x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 16) { gemm_microkernel_v_fp32_7x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 1) { gemm_microkernel_v_fp32_8x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 2) { gemm_microkernel_v_fp32_8x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 3) { gemm_microkernel_v_fp32_8x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 4) { gemm_microkernel_v_fp32_8x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 5) { gemm_microkernel_v_fp32_8x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 6) { gemm_microkernel_v_fp32_8x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 7) { gemm_microkernel_v_fp32_8x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 8) { gemm_microkernel_v_fp32_8x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 9) { gemm_microkernel_v_fp32_8x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 10) { gemm_microkernel_v_fp32_8x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 11) { gemm_microkernel_v_fp32_8x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 12) { gemm_microkernel_v_fp32_8x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 13) { gemm_microkernel_v_fp32_8x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 14) { gemm_microkernel_v_fp32_8x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 15) { gemm_microkernel_v_fp32_8x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 16) { gemm_microkernel_v_fp32_8x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 1) { gemm_microkernel_v_fp32_9x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 2) { gemm_microkernel_v_fp32_9x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 3) { gemm_microkernel_v_fp32_9x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 4) { gemm_microkernel_v_fp32_9x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 5) { gemm_microkernel_v_fp32_9x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 6) { gemm_microkernel_v_fp32_9x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 7) { gemm_microkernel_v_fp32_9x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 8) { gemm_microkernel_v_fp32_9x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 9) { gemm_microkernel_v_fp32_9x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 10) { gemm_microkernel_v_fp32_9x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 11) { gemm_microkernel_v_fp32_9x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 12) { gemm_microkernel_v_fp32_9x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 13) { gemm_microkernel_v_fp32_9x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 14) { gemm_microkernel_v_fp32_9x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 15) { gemm_microkernel_v_fp32_9x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 16) { gemm_microkernel_v_fp32_9x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 1) { gemm_microkernel_v_fp32_10x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 2) { gemm_microkernel_v_fp32_10x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 3) { gemm_microkernel_v_fp32_10x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 4) { gemm_microkernel_v_fp32_10x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 5) { gemm_microkernel_v_fp32_10x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 6) { gemm_microkernel_v_fp32_10x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 7) { gemm_microkernel_v_fp32_10x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 8) { gemm_microkernel_v_fp32_10x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 9) { gemm_microkernel_v_fp32_10x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 10) { gemm_microkernel_v_fp32_10x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 11) { gemm_microkernel_v_fp32_10x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 12) { gemm_microkernel_v_fp32_10x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 13) { gemm_microkernel_v_fp32_10x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 14) { gemm_microkernel_v_fp32_10x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 15) { gemm_microkernel_v_fp32_10x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 16) { gemm_microkernel_v_fp32_10x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 1) { gemm_microkernel_v_fp32_11x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 2) { gemm_microkernel_v_fp32_11x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 3) { gemm_microkernel_v_fp32_11x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 4) { gemm_microkernel_v_fp32_11x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 5) { gemm_microkernel_v_fp32_11x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 6) { gemm_microkernel_v_fp32_11x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 7) { gemm_microkernel_v_fp32_11x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 8) { gemm_microkernel_v_fp32_11x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 9) { gemm_microkernel_v_fp32_11x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 10) { gemm_microkernel_v_fp32_11x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 11) { gemm_microkernel_v_fp32_11x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 12) { gemm_microkernel_v_fp32_11x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 13) { gemm_microkernel_v_fp32_11x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 14) { gemm_microkernel_v_fp32_11x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 15) { gemm_microkernel_v_fp32_11x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 16) { gemm_microkernel_v_fp32_11x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 1) { gemm_microkernel_v_fp32_12x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 2) { gemm_microkernel_v_fp32_12x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 3) { gemm_microkernel_v_fp32_12x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 4) { gemm_microkernel_v_fp32_12x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 5) { gemm_microkernel_v_fp32_12x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 6) { gemm_microkernel_v_fp32_12x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 7) { gemm_microkernel_v_fp32_12x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 8) { gemm_microkernel_v_fp32_12x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 9) { gemm_microkernel_v_fp32_12x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 10) { gemm_microkernel_v_fp32_12x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 11) { gemm_microkernel_v_fp32_12x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 12) { gemm_microkernel_v_fp32_12x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 13) { gemm_microkernel_v_fp32_12x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 14) { gemm_microkernel_v_fp32_12x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 15) { gemm_microkernel_v_fp32_12x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 16) { gemm_microkernel_v_fp32_12x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 1) { gemm_microkernel_v_fp32_13x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 2) { gemm_microkernel_v_fp32_13x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 3) { gemm_microkernel_v_fp32_13x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 4) { gemm_microkernel_v_fp32_13x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 5) { gemm_microkernel_v_fp32_13x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 6) { gemm_microkernel_v_fp32_13x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 7) { gemm_microkernel_v_fp32_13x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 8) { gemm_microkernel_v_fp32_13x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 9) { gemm_microkernel_v_fp32_13x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 10) { gemm_microkernel_v_fp32_13x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 11) { gemm_microkernel_v_fp32_13x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 12) { gemm_microkernel_v_fp32_13x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 13) { gemm_microkernel_v_fp32_13x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 14) { gemm_microkernel_v_fp32_13x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 15) { gemm_microkernel_v_fp32_13x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 16) { gemm_microkernel_v_fp32_13x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 1) { gemm_microkernel_v_fp32_14x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 2) { gemm_microkernel_v_fp32_14x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 3) { gemm_microkernel_v_fp32_14x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 4) { gemm_microkernel_v_fp32_14x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 5) { gemm_microkernel_v_fp32_14x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 6) { gemm_microkernel_v_fp32_14x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 7) { gemm_microkernel_v_fp32_14x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 8) { gemm_microkernel_v_fp32_14x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 9) { gemm_microkernel_v_fp32_14x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 10) { gemm_microkernel_v_fp32_14x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 11) { gemm_microkernel_v_fp32_14x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 12) { gemm_microkernel_v_fp32_14x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 13) { gemm_microkernel_v_fp32_14x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 14) { gemm_microkernel_v_fp32_14x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 15) { gemm_microkernel_v_fp32_14x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 16) { gemm_microkernel_v_fp32_14x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 1) { gemm_microkernel_v_fp32_15x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 2) { gemm_microkernel_v_fp32_15x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 3) { gemm_microkernel_v_fp32_15x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 4) { gemm_microkernel_v_fp32_15x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 5) { gemm_microkernel_v_fp32_15x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 6) { gemm_microkernel_v_fp32_15x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 7) { gemm_microkernel_v_fp32_15x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 8) { gemm_microkernel_v_fp32_15x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 9) { gemm_microkernel_v_fp32_15x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 10) { gemm_microkernel_v_fp32_15x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 11) { gemm_microkernel_v_fp32_15x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 12) { gemm_microkernel_v_fp32_15x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 13) { gemm_microkernel_v_fp32_15x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 14) { gemm_microkernel_v_fp32_15x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 15) { gemm_microkernel_v_fp32_15x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 16) { gemm_microkernel_v_fp32_15x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 1) { gemm_microkernel_v_fp32_16x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 2) { gemm_microkernel_v_fp32_16x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 3) { gemm_microkernel_v_fp32_16x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 4) { gemm_microkernel_v_fp32_16x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 5) { gemm_microkernel_v_fp32_16x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 6) { gemm_microkernel_v_fp32_16x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 7) { gemm_microkernel_v_fp32_16x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 8) { gemm_microkernel_v_fp32_16x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 9) { gemm_microkernel_v_fp32_16x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 10) { gemm_microkernel_v_fp32_16x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 11) { gemm_microkernel_v_fp32_16x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 12) { gemm_microkernel_v_fp32_16x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 13) { gemm_microkernel_v_fp32_16x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 14) { gemm_microkernel_v_fp32_16x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 15) { gemm_microkernel_v_fp32_16x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 16) { gemm_microkernel_v_fp32_16x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 1) { gemm_microkernel_v_fp32_17x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 2) { gemm_microkernel_v_fp32_17x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 3) { gemm_microkernel_v_fp32_17x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 4) { gemm_microkernel_v_fp32_17x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 5) { gemm_microkernel_v_fp32_17x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 6) { gemm_microkernel_v_fp32_17x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 7) { gemm_microkernel_v_fp32_17x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 8) { gemm_microkernel_v_fp32_17x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 9) { gemm_microkernel_v_fp32_17x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 10) { gemm_microkernel_v_fp32_17x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 11) { gemm_microkernel_v_fp32_17x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 12) { gemm_microkernel_v_fp32_17x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 13) { gemm_microkernel_v_fp32_17x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 14) { gemm_microkernel_v_fp32_17x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 15) { gemm_microkernel_v_fp32_17x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 16) { gemm_microkernel_v_fp32_17x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 1) { gemm_microkernel_v_fp32_18x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 2) { gemm_microkernel_v_fp32_18x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 3) { gemm_microkernel_v_fp32_18x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 4) { gemm_microkernel_v_fp32_18x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 5) { gemm_microkernel_v_fp32_18x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 6) { gemm_microkernel_v_fp32_18x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 7) { gemm_microkernel_v_fp32_18x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 8) { gemm_microkernel_v_fp32_18x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 9) { gemm_microkernel_v_fp32_18x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 10) { gemm_microkernel_v_fp32_18x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 11) { gemm_microkernel_v_fp32_18x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 12) { gemm_microkernel_v_fp32_18x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 13) { gemm_microkernel_v_fp32_18x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 14) { gemm_microkernel_v_fp32_18x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 15) { gemm_microkernel_v_fp32_18x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 16) { gemm_microkernel_v_fp32_18x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 1) { gemm_microkernel_v_fp32_19x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 2) { gemm_microkernel_v_fp32_19x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 3) { gemm_microkernel_v_fp32_19x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 4) { gemm_microkernel_v_fp32_19x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 5) { gemm_microkernel_v_fp32_19x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 6) { gemm_microkernel_v_fp32_19x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 7) { gemm_microkernel_v_fp32_19x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 8) { gemm_microkernel_v_fp32_19x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 9) { gemm_microkernel_v_fp32_19x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 10) { gemm_microkernel_v_fp32_19x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 11) { gemm_microkernel_v_fp32_19x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 12) { gemm_microkernel_v_fp32_19x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 13) { gemm_microkernel_v_fp32_19x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 14) { gemm_microkernel_v_fp32_19x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 15) { gemm_microkernel_v_fp32_19x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 16) { gemm_microkernel_v_fp32_19x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 1) { gemm_microkernel_v_fp32_20x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 2) { gemm_microkernel_v_fp32_20x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 3) { gemm_microkernel_v_fp32_20x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 4) { gemm_microkernel_v_fp32_20x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 5) { gemm_microkernel_v_fp32_20x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 6) { gemm_microkernel_v_fp32_20x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 7) { gemm_microkernel_v_fp32_20x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 8) { gemm_microkernel_v_fp32_20x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 9) { gemm_microkernel_v_fp32_20x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 10) { gemm_microkernel_v_fp32_20x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 11) { gemm_microkernel_v_fp32_20x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 12) { gemm_microkernel_v_fp32_20x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 13) { gemm_microkernel_v_fp32_20x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 14) { gemm_microkernel_v_fp32_20x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 15) { gemm_microkernel_v_fp32_20x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 16) { gemm_microkernel_v_fp32_20x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 1) { gemm_microkernel_v_fp32_21x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 2) { gemm_microkernel_v_fp32_21x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 3) { gemm_microkernel_v_fp32_21x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 4) { gemm_microkernel_v_fp32_21x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 5) { gemm_microkernel_v_fp32_21x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 6) { gemm_microkernel_v_fp32_21x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 7) { gemm_microkernel_v_fp32_21x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 8) { gemm_microkernel_v_fp32_21x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 9) { gemm_microkernel_v_fp32_21x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 10) { gemm_microkernel_v_fp32_21x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 11) { gemm_microkernel_v_fp32_21x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 12) { gemm_microkernel_v_fp32_21x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 13) { gemm_microkernel_v_fp32_21x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 14) { gemm_microkernel_v_fp32_21x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 15) { gemm_microkernel_v_fp32_21x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 16) { gemm_microkernel_v_fp32_21x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 1) { gemm_microkernel_v_fp32_22x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 2) { gemm_microkernel_v_fp32_22x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 3) { gemm_microkernel_v_fp32_22x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 4) { gemm_microkernel_v_fp32_22x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 5) { gemm_microkernel_v_fp32_22x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 6) { gemm_microkernel_v_fp32_22x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 7) { gemm_microkernel_v_fp32_22x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 8) { gemm_microkernel_v_fp32_22x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 9) { gemm_microkernel_v_fp32_22x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 10) { gemm_microkernel_v_fp32_22x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 11) { gemm_microkernel_v_fp32_22x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 12) { gemm_microkernel_v_fp32_22x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 13) { gemm_microkernel_v_fp32_22x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 14) { gemm_microkernel_v_fp32_22x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 15) { gemm_microkernel_v_fp32_22x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 16) { gemm_microkernel_v_fp32_22x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 1) { gemm_microkernel_v_fp32_23x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 2) { gemm_microkernel_v_fp32_23x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 3) { gemm_microkernel_v_fp32_23x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 4) { gemm_microkernel_v_fp32_23x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 5) { gemm_microkernel_v_fp32_23x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 6) { gemm_microkernel_v_fp32_23x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 7) { gemm_microkernel_v_fp32_23x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 8) { gemm_microkernel_v_fp32_23x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 9) { gemm_microkernel_v_fp32_23x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 10) { gemm_microkernel_v_fp32_23x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 11) { gemm_microkernel_v_fp32_23x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 12) { gemm_microkernel_v_fp32_23x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 13) { gemm_microkernel_v_fp32_23x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 14) { gemm_microkernel_v_fp32_23x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 15) { gemm_microkernel_v_fp32_23x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 16) { gemm_microkernel_v_fp32_23x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 1) { gemm_microkernel_v_fp32_24x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 2) { gemm_microkernel_v_fp32_24x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 3) { gemm_microkernel_v_fp32_24x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 4) { gemm_microkernel_v_fp32_24x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 5) { gemm_microkernel_v_fp32_24x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 6) { gemm_microkernel_v_fp32_24x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 7) { gemm_microkernel_v_fp32_24x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 8) { gemm_microkernel_v_fp32_24x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 9) { gemm_microkernel_v_fp32_24x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 10) { gemm_microkernel_v_fp32_24x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 11) { gemm_microkernel_v_fp32_24x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 12) { gemm_microkernel_v_fp32_24x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 13) { gemm_microkernel_v_fp32_24x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 14) { gemm_microkernel_v_fp32_24x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 15) { gemm_microkernel_v_fp32_24x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 16) { gemm_microkernel_v_fp32_24x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 1) { gemm_microkernel_v_fp32_25x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 2) { gemm_microkernel_v_fp32_25x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 3) { gemm_microkernel_v_fp32_25x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 4) { gemm_microkernel_v_fp32_25x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 5) { gemm_microkernel_v_fp32_25x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 6) { gemm_microkernel_v_fp32_25x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 7) { gemm_microkernel_v_fp32_25x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 8) { gemm_microkernel_v_fp32_25x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 9) { gemm_microkernel_v_fp32_25x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 10) { gemm_microkernel_v_fp32_25x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 11) { gemm_microkernel_v_fp32_25x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 12) { gemm_microkernel_v_fp32_25x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 13) { gemm_microkernel_v_fp32_25x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 14) { gemm_microkernel_v_fp32_25x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 15) { gemm_microkernel_v_fp32_25x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 16) { gemm_microkernel_v_fp32_25x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 1) { gemm_microkernel_v_fp32_26x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 2) { gemm_microkernel_v_fp32_26x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 3) { gemm_microkernel_v_fp32_26x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 4) { gemm_microkernel_v_fp32_26x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 5) { gemm_microkernel_v_fp32_26x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 6) { gemm_microkernel_v_fp32_26x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 7) { gemm_microkernel_v_fp32_26x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 8) { gemm_microkernel_v_fp32_26x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 9) { gemm_microkernel_v_fp32_26x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 10) { gemm_microkernel_v_fp32_26x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 11) { gemm_microkernel_v_fp32_26x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 12) { gemm_microkernel_v_fp32_26x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 13) { gemm_microkernel_v_fp32_26x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 14) { gemm_microkernel_v_fp32_26x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 15) { gemm_microkernel_v_fp32_26x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 16) { gemm_microkernel_v_fp32_26x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 1) { gemm_microkernel_v_fp32_27x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 2) { gemm_microkernel_v_fp32_27x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 3) { gemm_microkernel_v_fp32_27x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 4) { gemm_microkernel_v_fp32_27x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 5) { gemm_microkernel_v_fp32_27x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 6) { gemm_microkernel_v_fp32_27x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 7) { gemm_microkernel_v_fp32_27x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 8) { gemm_microkernel_v_fp32_27x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 9) { gemm_microkernel_v_fp32_27x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 10) { gemm_microkernel_v_fp32_27x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 11) { gemm_microkernel_v_fp32_27x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 12) { gemm_microkernel_v_fp32_27x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 13) { gemm_microkernel_v_fp32_27x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 14) { gemm_microkernel_v_fp32_27x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 15) { gemm_microkernel_v_fp32_27x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 16) { gemm_microkernel_v_fp32_27x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 1) { gemm_microkernel_v_fp32_28x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 2) { gemm_microkernel_v_fp32_28x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 3) { gemm_microkernel_v_fp32_28x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 4) { gemm_microkernel_v_fp32_28x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 5) { gemm_microkernel_v_fp32_28x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 6) { gemm_microkernel_v_fp32_28x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 7) { gemm_microkernel_v_fp32_28x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 8) { gemm_microkernel_v_fp32_28x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 9) { gemm_microkernel_v_fp32_28x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 10) { gemm_microkernel_v_fp32_28x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 11) { gemm_microkernel_v_fp32_28x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 12) { gemm_microkernel_v_fp32_28x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 13) { gemm_microkernel_v_fp32_28x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 14) { gemm_microkernel_v_fp32_28x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 15) { gemm_microkernel_v_fp32_28x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 16) { gemm_microkernel_v_fp32_28x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 1) { gemm_microkernel_v_fp32_29x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 2) { gemm_microkernel_v_fp32_29x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 3) { gemm_microkernel_v_fp32_29x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 4) { gemm_microkernel_v_fp32_29x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 5) { gemm_microkernel_v_fp32_29x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 6) { gemm_microkernel_v_fp32_29x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 7) { gemm_microkernel_v_fp32_29x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 8) { gemm_microkernel_v_fp32_29x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 9) { gemm_microkernel_v_fp32_29x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 10) { gemm_microkernel_v_fp32_29x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 11) { gemm_microkernel_v_fp32_29x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 12) { gemm_microkernel_v_fp32_29x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 13) { gemm_microkernel_v_fp32_29x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 14) { gemm_microkernel_v_fp32_29x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 15) { gemm_microkernel_v_fp32_29x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 16) { gemm_microkernel_v_fp32_29x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 1) { gemm_microkernel_v_fp32_30x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 2) { gemm_microkernel_v_fp32_30x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 3) { gemm_microkernel_v_fp32_30x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 4) { gemm_microkernel_v_fp32_30x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 5) { gemm_microkernel_v_fp32_30x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 6) { gemm_microkernel_v_fp32_30x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 7) { gemm_microkernel_v_fp32_30x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 8) { gemm_microkernel_v_fp32_30x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 9) { gemm_microkernel_v_fp32_30x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 10) { gemm_microkernel_v_fp32_30x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 11) { gemm_microkernel_v_fp32_30x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 12) { gemm_microkernel_v_fp32_30x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 13) { gemm_microkernel_v_fp32_30x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 14) { gemm_microkernel_v_fp32_30x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 15) { gemm_microkernel_v_fp32_30x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 16) { gemm_microkernel_v_fp32_30x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 1) { gemm_microkernel_v_fp32_31x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 2) { gemm_microkernel_v_fp32_31x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 3) { gemm_microkernel_v_fp32_31x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 4) { gemm_microkernel_v_fp32_31x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 5) { gemm_microkernel_v_fp32_31x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 6) { gemm_microkernel_v_fp32_31x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 7) { gemm_microkernel_v_fp32_31x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 8) { gemm_microkernel_v_fp32_31x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 9) { gemm_microkernel_v_fp32_31x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 10) { gemm_microkernel_v_fp32_31x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 11) { gemm_microkernel_v_fp32_31x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 12) { gemm_microkernel_v_fp32_31x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 13) { gemm_microkernel_v_fp32_31x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 14) { gemm_microkernel_v_fp32_31x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 15) { gemm_microkernel_v_fp32_31x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 16) { gemm_microkernel_v_fp32_31x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 1) { gemm_microkernel_v_fp32_32x1_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 2) { gemm_microkernel_v_fp32_32x2_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 3) { gemm_microkernel_v_fp32_32x3_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 4) { gemm_microkernel_v_fp32_32x4_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 5) { gemm_microkernel_v_fp32_32x5_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 6) { gemm_microkernel_v_fp32_32x6_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 7) { gemm_microkernel_v_fp32_32x7_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 8) { gemm_microkernel_v_fp32_32x8_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 9) { gemm_microkernel_v_fp32_32x9_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 10) { gemm_microkernel_v_fp32_32x10_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 11) { gemm_microkernel_v_fp32_32x11_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 12) { gemm_microkernel_v_fp32_32x12_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 13) { gemm_microkernel_v_fp32_32x13_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 14) { gemm_microkernel_v_fp32_32x14_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 15) { gemm_microkernel_v_fp32_32x15_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 16) { gemm_microkernel_v_fp32_32x16_b0(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for cur_mr=%zu cur_nr=%zu\n", cur_mr, cur_nr); abort();
    } else {
        if (cur_mr == 1 && cur_nr == 1) { gemm_microkernel_v_fp32_1x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 2) { gemm_microkernel_v_fp32_1x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 3) { gemm_microkernel_v_fp32_1x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 4) { gemm_microkernel_v_fp32_1x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 5) { gemm_microkernel_v_fp32_1x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 6) { gemm_microkernel_v_fp32_1x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 7) { gemm_microkernel_v_fp32_1x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 8) { gemm_microkernel_v_fp32_1x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 9) { gemm_microkernel_v_fp32_1x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 10) { gemm_microkernel_v_fp32_1x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 11) { gemm_microkernel_v_fp32_1x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 12) { gemm_microkernel_v_fp32_1x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 13) { gemm_microkernel_v_fp32_1x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 14) { gemm_microkernel_v_fp32_1x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 15) { gemm_microkernel_v_fp32_1x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 1 && cur_nr == 16) { gemm_microkernel_v_fp32_1x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 1) { gemm_microkernel_v_fp32_2x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 2) { gemm_microkernel_v_fp32_2x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 3) { gemm_microkernel_v_fp32_2x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 4) { gemm_microkernel_v_fp32_2x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 5) { gemm_microkernel_v_fp32_2x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 6) { gemm_microkernel_v_fp32_2x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 7) { gemm_microkernel_v_fp32_2x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 8) { gemm_microkernel_v_fp32_2x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 9) { gemm_microkernel_v_fp32_2x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 10) { gemm_microkernel_v_fp32_2x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 11) { gemm_microkernel_v_fp32_2x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 12) { gemm_microkernel_v_fp32_2x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 13) { gemm_microkernel_v_fp32_2x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 14) { gemm_microkernel_v_fp32_2x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 15) { gemm_microkernel_v_fp32_2x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 2 && cur_nr == 16) { gemm_microkernel_v_fp32_2x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 1) { gemm_microkernel_v_fp32_3x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 2) { gemm_microkernel_v_fp32_3x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 3) { gemm_microkernel_v_fp32_3x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 4) { gemm_microkernel_v_fp32_3x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 5) { gemm_microkernel_v_fp32_3x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 6) { gemm_microkernel_v_fp32_3x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 7) { gemm_microkernel_v_fp32_3x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 8) { gemm_microkernel_v_fp32_3x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 9) { gemm_microkernel_v_fp32_3x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 10) { gemm_microkernel_v_fp32_3x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 11) { gemm_microkernel_v_fp32_3x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 12) { gemm_microkernel_v_fp32_3x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 13) { gemm_microkernel_v_fp32_3x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 14) { gemm_microkernel_v_fp32_3x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 15) { gemm_microkernel_v_fp32_3x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 3 && cur_nr == 16) { gemm_microkernel_v_fp32_3x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 1) { gemm_microkernel_v_fp32_4x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 2) { gemm_microkernel_v_fp32_4x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 3) { gemm_microkernel_v_fp32_4x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 4) { gemm_microkernel_v_fp32_4x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 5) { gemm_microkernel_v_fp32_4x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 6) { gemm_microkernel_v_fp32_4x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 7) { gemm_microkernel_v_fp32_4x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 8) { gemm_microkernel_v_fp32_4x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 9) { gemm_microkernel_v_fp32_4x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 10) { gemm_microkernel_v_fp32_4x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 11) { gemm_microkernel_v_fp32_4x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 12) { gemm_microkernel_v_fp32_4x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 13) { gemm_microkernel_v_fp32_4x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 14) { gemm_microkernel_v_fp32_4x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 15) { gemm_microkernel_v_fp32_4x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 4 && cur_nr == 16) { gemm_microkernel_v_fp32_4x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 1) { gemm_microkernel_v_fp32_5x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 2) { gemm_microkernel_v_fp32_5x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 3) { gemm_microkernel_v_fp32_5x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 4) { gemm_microkernel_v_fp32_5x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 5) { gemm_microkernel_v_fp32_5x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 6) { gemm_microkernel_v_fp32_5x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 7) { gemm_microkernel_v_fp32_5x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 8) { gemm_microkernel_v_fp32_5x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 9) { gemm_microkernel_v_fp32_5x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 10) { gemm_microkernel_v_fp32_5x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 11) { gemm_microkernel_v_fp32_5x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 12) { gemm_microkernel_v_fp32_5x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 13) { gemm_microkernel_v_fp32_5x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 14) { gemm_microkernel_v_fp32_5x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 15) { gemm_microkernel_v_fp32_5x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 5 && cur_nr == 16) { gemm_microkernel_v_fp32_5x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 1) { gemm_microkernel_v_fp32_6x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 2) { gemm_microkernel_v_fp32_6x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 3) { gemm_microkernel_v_fp32_6x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 4) { gemm_microkernel_v_fp32_6x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 5) { gemm_microkernel_v_fp32_6x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 6) { gemm_microkernel_v_fp32_6x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 7) { gemm_microkernel_v_fp32_6x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 8) { gemm_microkernel_v_fp32_6x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 9) { gemm_microkernel_v_fp32_6x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 10) { gemm_microkernel_v_fp32_6x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 11) { gemm_microkernel_v_fp32_6x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 12) { gemm_microkernel_v_fp32_6x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 13) { gemm_microkernel_v_fp32_6x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 14) { gemm_microkernel_v_fp32_6x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 15) { gemm_microkernel_v_fp32_6x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 6 && cur_nr == 16) { gemm_microkernel_v_fp32_6x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 1) { gemm_microkernel_v_fp32_7x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 2) { gemm_microkernel_v_fp32_7x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 3) { gemm_microkernel_v_fp32_7x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 4) { gemm_microkernel_v_fp32_7x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 5) { gemm_microkernel_v_fp32_7x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 6) { gemm_microkernel_v_fp32_7x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 7) { gemm_microkernel_v_fp32_7x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 8) { gemm_microkernel_v_fp32_7x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 9) { gemm_microkernel_v_fp32_7x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 10) { gemm_microkernel_v_fp32_7x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 11) { gemm_microkernel_v_fp32_7x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 12) { gemm_microkernel_v_fp32_7x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 13) { gemm_microkernel_v_fp32_7x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 14) { gemm_microkernel_v_fp32_7x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 15) { gemm_microkernel_v_fp32_7x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 7 && cur_nr == 16) { gemm_microkernel_v_fp32_7x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 1) { gemm_microkernel_v_fp32_8x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 2) { gemm_microkernel_v_fp32_8x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 3) { gemm_microkernel_v_fp32_8x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 4) { gemm_microkernel_v_fp32_8x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 5) { gemm_microkernel_v_fp32_8x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 6) { gemm_microkernel_v_fp32_8x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 7) { gemm_microkernel_v_fp32_8x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 8) { gemm_microkernel_v_fp32_8x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 9) { gemm_microkernel_v_fp32_8x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 10) { gemm_microkernel_v_fp32_8x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 11) { gemm_microkernel_v_fp32_8x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 12) { gemm_microkernel_v_fp32_8x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 13) { gemm_microkernel_v_fp32_8x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 14) { gemm_microkernel_v_fp32_8x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 15) { gemm_microkernel_v_fp32_8x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 8 && cur_nr == 16) { gemm_microkernel_v_fp32_8x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 1) { gemm_microkernel_v_fp32_9x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 2) { gemm_microkernel_v_fp32_9x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 3) { gemm_microkernel_v_fp32_9x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 4) { gemm_microkernel_v_fp32_9x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 5) { gemm_microkernel_v_fp32_9x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 6) { gemm_microkernel_v_fp32_9x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 7) { gemm_microkernel_v_fp32_9x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 8) { gemm_microkernel_v_fp32_9x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 9) { gemm_microkernel_v_fp32_9x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 10) { gemm_microkernel_v_fp32_9x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 11) { gemm_microkernel_v_fp32_9x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 12) { gemm_microkernel_v_fp32_9x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 13) { gemm_microkernel_v_fp32_9x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 14) { gemm_microkernel_v_fp32_9x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 15) { gemm_microkernel_v_fp32_9x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 9 && cur_nr == 16) { gemm_microkernel_v_fp32_9x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 1) { gemm_microkernel_v_fp32_10x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 2) { gemm_microkernel_v_fp32_10x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 3) { gemm_microkernel_v_fp32_10x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 4) { gemm_microkernel_v_fp32_10x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 5) { gemm_microkernel_v_fp32_10x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 6) { gemm_microkernel_v_fp32_10x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 7) { gemm_microkernel_v_fp32_10x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 8) { gemm_microkernel_v_fp32_10x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 9) { gemm_microkernel_v_fp32_10x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 10) { gemm_microkernel_v_fp32_10x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 11) { gemm_microkernel_v_fp32_10x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 12) { gemm_microkernel_v_fp32_10x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 13) { gemm_microkernel_v_fp32_10x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 14) { gemm_microkernel_v_fp32_10x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 15) { gemm_microkernel_v_fp32_10x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 10 && cur_nr == 16) { gemm_microkernel_v_fp32_10x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 1) { gemm_microkernel_v_fp32_11x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 2) { gemm_microkernel_v_fp32_11x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 3) { gemm_microkernel_v_fp32_11x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 4) { gemm_microkernel_v_fp32_11x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 5) { gemm_microkernel_v_fp32_11x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 6) { gemm_microkernel_v_fp32_11x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 7) { gemm_microkernel_v_fp32_11x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 8) { gemm_microkernel_v_fp32_11x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 9) { gemm_microkernel_v_fp32_11x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 10) { gemm_microkernel_v_fp32_11x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 11) { gemm_microkernel_v_fp32_11x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 12) { gemm_microkernel_v_fp32_11x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 13) { gemm_microkernel_v_fp32_11x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 14) { gemm_microkernel_v_fp32_11x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 15) { gemm_microkernel_v_fp32_11x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 11 && cur_nr == 16) { gemm_microkernel_v_fp32_11x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 1) { gemm_microkernel_v_fp32_12x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 2) { gemm_microkernel_v_fp32_12x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 3) { gemm_microkernel_v_fp32_12x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 4) { gemm_microkernel_v_fp32_12x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 5) { gemm_microkernel_v_fp32_12x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 6) { gemm_microkernel_v_fp32_12x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 7) { gemm_microkernel_v_fp32_12x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 8) { gemm_microkernel_v_fp32_12x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 9) { gemm_microkernel_v_fp32_12x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 10) { gemm_microkernel_v_fp32_12x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 11) { gemm_microkernel_v_fp32_12x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 12) { gemm_microkernel_v_fp32_12x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 13) { gemm_microkernel_v_fp32_12x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 14) { gemm_microkernel_v_fp32_12x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 15) { gemm_microkernel_v_fp32_12x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 12 && cur_nr == 16) { gemm_microkernel_v_fp32_12x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 1) { gemm_microkernel_v_fp32_13x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 2) { gemm_microkernel_v_fp32_13x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 3) { gemm_microkernel_v_fp32_13x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 4) { gemm_microkernel_v_fp32_13x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 5) { gemm_microkernel_v_fp32_13x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 6) { gemm_microkernel_v_fp32_13x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 7) { gemm_microkernel_v_fp32_13x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 8) { gemm_microkernel_v_fp32_13x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 9) { gemm_microkernel_v_fp32_13x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 10) { gemm_microkernel_v_fp32_13x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 11) { gemm_microkernel_v_fp32_13x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 12) { gemm_microkernel_v_fp32_13x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 13) { gemm_microkernel_v_fp32_13x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 14) { gemm_microkernel_v_fp32_13x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 15) { gemm_microkernel_v_fp32_13x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 13 && cur_nr == 16) { gemm_microkernel_v_fp32_13x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 1) { gemm_microkernel_v_fp32_14x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 2) { gemm_microkernel_v_fp32_14x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 3) { gemm_microkernel_v_fp32_14x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 4) { gemm_microkernel_v_fp32_14x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 5) { gemm_microkernel_v_fp32_14x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 6) { gemm_microkernel_v_fp32_14x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 7) { gemm_microkernel_v_fp32_14x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 8) { gemm_microkernel_v_fp32_14x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 9) { gemm_microkernel_v_fp32_14x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 10) { gemm_microkernel_v_fp32_14x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 11) { gemm_microkernel_v_fp32_14x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 12) { gemm_microkernel_v_fp32_14x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 13) { gemm_microkernel_v_fp32_14x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 14) { gemm_microkernel_v_fp32_14x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 15) { gemm_microkernel_v_fp32_14x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 14 && cur_nr == 16) { gemm_microkernel_v_fp32_14x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 1) { gemm_microkernel_v_fp32_15x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 2) { gemm_microkernel_v_fp32_15x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 3) { gemm_microkernel_v_fp32_15x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 4) { gemm_microkernel_v_fp32_15x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 5) { gemm_microkernel_v_fp32_15x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 6) { gemm_microkernel_v_fp32_15x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 7) { gemm_microkernel_v_fp32_15x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 8) { gemm_microkernel_v_fp32_15x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 9) { gemm_microkernel_v_fp32_15x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 10) { gemm_microkernel_v_fp32_15x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 11) { gemm_microkernel_v_fp32_15x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 12) { gemm_microkernel_v_fp32_15x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 13) { gemm_microkernel_v_fp32_15x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 14) { gemm_microkernel_v_fp32_15x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 15) { gemm_microkernel_v_fp32_15x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 15 && cur_nr == 16) { gemm_microkernel_v_fp32_15x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 1) { gemm_microkernel_v_fp32_16x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 2) { gemm_microkernel_v_fp32_16x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 3) { gemm_microkernel_v_fp32_16x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 4) { gemm_microkernel_v_fp32_16x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 5) { gemm_microkernel_v_fp32_16x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 6) { gemm_microkernel_v_fp32_16x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 7) { gemm_microkernel_v_fp32_16x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 8) { gemm_microkernel_v_fp32_16x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 9) { gemm_microkernel_v_fp32_16x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 10) { gemm_microkernel_v_fp32_16x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 11) { gemm_microkernel_v_fp32_16x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 12) { gemm_microkernel_v_fp32_16x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 13) { gemm_microkernel_v_fp32_16x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 14) { gemm_microkernel_v_fp32_16x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 15) { gemm_microkernel_v_fp32_16x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 16 && cur_nr == 16) { gemm_microkernel_v_fp32_16x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 1) { gemm_microkernel_v_fp32_17x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 2) { gemm_microkernel_v_fp32_17x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 3) { gemm_microkernel_v_fp32_17x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 4) { gemm_microkernel_v_fp32_17x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 5) { gemm_microkernel_v_fp32_17x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 6) { gemm_microkernel_v_fp32_17x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 7) { gemm_microkernel_v_fp32_17x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 8) { gemm_microkernel_v_fp32_17x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 9) { gemm_microkernel_v_fp32_17x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 10) { gemm_microkernel_v_fp32_17x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 11) { gemm_microkernel_v_fp32_17x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 12) { gemm_microkernel_v_fp32_17x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 13) { gemm_microkernel_v_fp32_17x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 14) { gemm_microkernel_v_fp32_17x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 15) { gemm_microkernel_v_fp32_17x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 17 && cur_nr == 16) { gemm_microkernel_v_fp32_17x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 1) { gemm_microkernel_v_fp32_18x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 2) { gemm_microkernel_v_fp32_18x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 3) { gemm_microkernel_v_fp32_18x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 4) { gemm_microkernel_v_fp32_18x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 5) { gemm_microkernel_v_fp32_18x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 6) { gemm_microkernel_v_fp32_18x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 7) { gemm_microkernel_v_fp32_18x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 8) { gemm_microkernel_v_fp32_18x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 9) { gemm_microkernel_v_fp32_18x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 10) { gemm_microkernel_v_fp32_18x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 11) { gemm_microkernel_v_fp32_18x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 12) { gemm_microkernel_v_fp32_18x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 13) { gemm_microkernel_v_fp32_18x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 14) { gemm_microkernel_v_fp32_18x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 15) { gemm_microkernel_v_fp32_18x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 18 && cur_nr == 16) { gemm_microkernel_v_fp32_18x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 1) { gemm_microkernel_v_fp32_19x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 2) { gemm_microkernel_v_fp32_19x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 3) { gemm_microkernel_v_fp32_19x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 4) { gemm_microkernel_v_fp32_19x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 5) { gemm_microkernel_v_fp32_19x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 6) { gemm_microkernel_v_fp32_19x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 7) { gemm_microkernel_v_fp32_19x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 8) { gemm_microkernel_v_fp32_19x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 9) { gemm_microkernel_v_fp32_19x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 10) { gemm_microkernel_v_fp32_19x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 11) { gemm_microkernel_v_fp32_19x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 12) { gemm_microkernel_v_fp32_19x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 13) { gemm_microkernel_v_fp32_19x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 14) { gemm_microkernel_v_fp32_19x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 15) { gemm_microkernel_v_fp32_19x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 19 && cur_nr == 16) { gemm_microkernel_v_fp32_19x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 1) { gemm_microkernel_v_fp32_20x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 2) { gemm_microkernel_v_fp32_20x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 3) { gemm_microkernel_v_fp32_20x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 4) { gemm_microkernel_v_fp32_20x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 5) { gemm_microkernel_v_fp32_20x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 6) { gemm_microkernel_v_fp32_20x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 7) { gemm_microkernel_v_fp32_20x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 8) { gemm_microkernel_v_fp32_20x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 9) { gemm_microkernel_v_fp32_20x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 10) { gemm_microkernel_v_fp32_20x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 11) { gemm_microkernel_v_fp32_20x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 12) { gemm_microkernel_v_fp32_20x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 13) { gemm_microkernel_v_fp32_20x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 14) { gemm_microkernel_v_fp32_20x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 15) { gemm_microkernel_v_fp32_20x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 20 && cur_nr == 16) { gemm_microkernel_v_fp32_20x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 1) { gemm_microkernel_v_fp32_21x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 2) { gemm_microkernel_v_fp32_21x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 3) { gemm_microkernel_v_fp32_21x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 4) { gemm_microkernel_v_fp32_21x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 5) { gemm_microkernel_v_fp32_21x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 6) { gemm_microkernel_v_fp32_21x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 7) { gemm_microkernel_v_fp32_21x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 8) { gemm_microkernel_v_fp32_21x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 9) { gemm_microkernel_v_fp32_21x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 10) { gemm_microkernel_v_fp32_21x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 11) { gemm_microkernel_v_fp32_21x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 12) { gemm_microkernel_v_fp32_21x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 13) { gemm_microkernel_v_fp32_21x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 14) { gemm_microkernel_v_fp32_21x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 15) { gemm_microkernel_v_fp32_21x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 21 && cur_nr == 16) { gemm_microkernel_v_fp32_21x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 1) { gemm_microkernel_v_fp32_22x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 2) { gemm_microkernel_v_fp32_22x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 3) { gemm_microkernel_v_fp32_22x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 4) { gemm_microkernel_v_fp32_22x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 5) { gemm_microkernel_v_fp32_22x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 6) { gemm_microkernel_v_fp32_22x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 7) { gemm_microkernel_v_fp32_22x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 8) { gemm_microkernel_v_fp32_22x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 9) { gemm_microkernel_v_fp32_22x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 10) { gemm_microkernel_v_fp32_22x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 11) { gemm_microkernel_v_fp32_22x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 12) { gemm_microkernel_v_fp32_22x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 13) { gemm_microkernel_v_fp32_22x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 14) { gemm_microkernel_v_fp32_22x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 15) { gemm_microkernel_v_fp32_22x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 22 && cur_nr == 16) { gemm_microkernel_v_fp32_22x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 1) { gemm_microkernel_v_fp32_23x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 2) { gemm_microkernel_v_fp32_23x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 3) { gemm_microkernel_v_fp32_23x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 4) { gemm_microkernel_v_fp32_23x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 5) { gemm_microkernel_v_fp32_23x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 6) { gemm_microkernel_v_fp32_23x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 7) { gemm_microkernel_v_fp32_23x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 8) { gemm_microkernel_v_fp32_23x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 9) { gemm_microkernel_v_fp32_23x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 10) { gemm_microkernel_v_fp32_23x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 11) { gemm_microkernel_v_fp32_23x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 12) { gemm_microkernel_v_fp32_23x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 13) { gemm_microkernel_v_fp32_23x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 14) { gemm_microkernel_v_fp32_23x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 15) { gemm_microkernel_v_fp32_23x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 23 && cur_nr == 16) { gemm_microkernel_v_fp32_23x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 1) { gemm_microkernel_v_fp32_24x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 2) { gemm_microkernel_v_fp32_24x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 3) { gemm_microkernel_v_fp32_24x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 4) { gemm_microkernel_v_fp32_24x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 5) { gemm_microkernel_v_fp32_24x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 6) { gemm_microkernel_v_fp32_24x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 7) { gemm_microkernel_v_fp32_24x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 8) { gemm_microkernel_v_fp32_24x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 9) { gemm_microkernel_v_fp32_24x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 10) { gemm_microkernel_v_fp32_24x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 11) { gemm_microkernel_v_fp32_24x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 12) { gemm_microkernel_v_fp32_24x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 13) { gemm_microkernel_v_fp32_24x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 14) { gemm_microkernel_v_fp32_24x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 15) { gemm_microkernel_v_fp32_24x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 24 && cur_nr == 16) { gemm_microkernel_v_fp32_24x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 1) { gemm_microkernel_v_fp32_25x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 2) { gemm_microkernel_v_fp32_25x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 3) { gemm_microkernel_v_fp32_25x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 4) { gemm_microkernel_v_fp32_25x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 5) { gemm_microkernel_v_fp32_25x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 6) { gemm_microkernel_v_fp32_25x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 7) { gemm_microkernel_v_fp32_25x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 8) { gemm_microkernel_v_fp32_25x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 9) { gemm_microkernel_v_fp32_25x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 10) { gemm_microkernel_v_fp32_25x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 11) { gemm_microkernel_v_fp32_25x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 12) { gemm_microkernel_v_fp32_25x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 13) { gemm_microkernel_v_fp32_25x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 14) { gemm_microkernel_v_fp32_25x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 15) { gemm_microkernel_v_fp32_25x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 25 && cur_nr == 16) { gemm_microkernel_v_fp32_25x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 1) { gemm_microkernel_v_fp32_26x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 2) { gemm_microkernel_v_fp32_26x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 3) { gemm_microkernel_v_fp32_26x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 4) { gemm_microkernel_v_fp32_26x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 5) { gemm_microkernel_v_fp32_26x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 6) { gemm_microkernel_v_fp32_26x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 7) { gemm_microkernel_v_fp32_26x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 8) { gemm_microkernel_v_fp32_26x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 9) { gemm_microkernel_v_fp32_26x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 10) { gemm_microkernel_v_fp32_26x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 11) { gemm_microkernel_v_fp32_26x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 12) { gemm_microkernel_v_fp32_26x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 13) { gemm_microkernel_v_fp32_26x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 14) { gemm_microkernel_v_fp32_26x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 15) { gemm_microkernel_v_fp32_26x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 26 && cur_nr == 16) { gemm_microkernel_v_fp32_26x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 1) { gemm_microkernel_v_fp32_27x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 2) { gemm_microkernel_v_fp32_27x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 3) { gemm_microkernel_v_fp32_27x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 4) { gemm_microkernel_v_fp32_27x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 5) { gemm_microkernel_v_fp32_27x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 6) { gemm_microkernel_v_fp32_27x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 7) { gemm_microkernel_v_fp32_27x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 8) { gemm_microkernel_v_fp32_27x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 9) { gemm_microkernel_v_fp32_27x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 10) { gemm_microkernel_v_fp32_27x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 11) { gemm_microkernel_v_fp32_27x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 12) { gemm_microkernel_v_fp32_27x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 13) { gemm_microkernel_v_fp32_27x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 14) { gemm_microkernel_v_fp32_27x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 15) { gemm_microkernel_v_fp32_27x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 27 && cur_nr == 16) { gemm_microkernel_v_fp32_27x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 1) { gemm_microkernel_v_fp32_28x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 2) { gemm_microkernel_v_fp32_28x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 3) { gemm_microkernel_v_fp32_28x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 4) { gemm_microkernel_v_fp32_28x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 5) { gemm_microkernel_v_fp32_28x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 6) { gemm_microkernel_v_fp32_28x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 7) { gemm_microkernel_v_fp32_28x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 8) { gemm_microkernel_v_fp32_28x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 9) { gemm_microkernel_v_fp32_28x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 10) { gemm_microkernel_v_fp32_28x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 11) { gemm_microkernel_v_fp32_28x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 12) { gemm_microkernel_v_fp32_28x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 13) { gemm_microkernel_v_fp32_28x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 14) { gemm_microkernel_v_fp32_28x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 15) { gemm_microkernel_v_fp32_28x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 28 && cur_nr == 16) { gemm_microkernel_v_fp32_28x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 1) { gemm_microkernel_v_fp32_29x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 2) { gemm_microkernel_v_fp32_29x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 3) { gemm_microkernel_v_fp32_29x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 4) { gemm_microkernel_v_fp32_29x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 5) { gemm_microkernel_v_fp32_29x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 6) { gemm_microkernel_v_fp32_29x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 7) { gemm_microkernel_v_fp32_29x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 8) { gemm_microkernel_v_fp32_29x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 9) { gemm_microkernel_v_fp32_29x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 10) { gemm_microkernel_v_fp32_29x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 11) { gemm_microkernel_v_fp32_29x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 12) { gemm_microkernel_v_fp32_29x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 13) { gemm_microkernel_v_fp32_29x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 14) { gemm_microkernel_v_fp32_29x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 15) { gemm_microkernel_v_fp32_29x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 29 && cur_nr == 16) { gemm_microkernel_v_fp32_29x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 1) { gemm_microkernel_v_fp32_30x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 2) { gemm_microkernel_v_fp32_30x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 3) { gemm_microkernel_v_fp32_30x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 4) { gemm_microkernel_v_fp32_30x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 5) { gemm_microkernel_v_fp32_30x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 6) { gemm_microkernel_v_fp32_30x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 7) { gemm_microkernel_v_fp32_30x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 8) { gemm_microkernel_v_fp32_30x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 9) { gemm_microkernel_v_fp32_30x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 10) { gemm_microkernel_v_fp32_30x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 11) { gemm_microkernel_v_fp32_30x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 12) { gemm_microkernel_v_fp32_30x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 13) { gemm_microkernel_v_fp32_30x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 14) { gemm_microkernel_v_fp32_30x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 15) { gemm_microkernel_v_fp32_30x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 30 && cur_nr == 16) { gemm_microkernel_v_fp32_30x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 1) { gemm_microkernel_v_fp32_31x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 2) { gemm_microkernel_v_fp32_31x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 3) { gemm_microkernel_v_fp32_31x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 4) { gemm_microkernel_v_fp32_31x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 5) { gemm_microkernel_v_fp32_31x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 6) { gemm_microkernel_v_fp32_31x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 7) { gemm_microkernel_v_fp32_31x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 8) { gemm_microkernel_v_fp32_31x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 9) { gemm_microkernel_v_fp32_31x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 10) { gemm_microkernel_v_fp32_31x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 11) { gemm_microkernel_v_fp32_31x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 12) { gemm_microkernel_v_fp32_31x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 13) { gemm_microkernel_v_fp32_31x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 14) { gemm_microkernel_v_fp32_31x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 15) { gemm_microkernel_v_fp32_31x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 31 && cur_nr == 16) { gemm_microkernel_v_fp32_31x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 1) { gemm_microkernel_v_fp32_32x1_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 2) { gemm_microkernel_v_fp32_32x2_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 3) { gemm_microkernel_v_fp32_32x3_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 4) { gemm_microkernel_v_fp32_32x4_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 5) { gemm_microkernel_v_fp32_32x5_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 6) { gemm_microkernel_v_fp32_32x6_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 7) { gemm_microkernel_v_fp32_32x7_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 8) { gemm_microkernel_v_fp32_32x8_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 9) { gemm_microkernel_v_fp32_32x9_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 10) { gemm_microkernel_v_fp32_32x10_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 11) { gemm_microkernel_v_fp32_32x11_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 12) { gemm_microkernel_v_fp32_32x12_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 13) { gemm_microkernel_v_fp32_32x13_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 14) { gemm_microkernel_v_fp32_32x14_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 15) { gemm_microkernel_v_fp32_32x15_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        if (cur_mr == 32 && cur_nr == 16) { gemm_microkernel_v_fp32_32x16_b1(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for cur_mr=%zu cur_nr=%zu\n", cur_mr, cur_nr); abort();
    }
}

void gemm_blis_generic(int32_t m, int32_t n, int32_t k, float alpha, float* A, int32_t lda, float* B, int32_t ldb, float beta, float* C, int32_t ldc, float* Ac, float* Bc, int32_t mc, int32_t nc, int32_t kc, int32_t mr, int32_t nr) {
    typedef void (*pack_CB_fn)(float*, float*, size_t, size_t, size_t);
    typedef void (*pack_RB_fn)(float*, float*, size_t, size_t, size_t);
    
    static const pack_CB_fn pack_CB_funcs[] = {
        nullptr,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR1,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR2,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR3,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR4,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR5,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR6,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR7,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR8,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR9,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR10,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR11,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR12,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR13,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR14,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR15,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR16,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR17,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR18,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR19,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR20,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR21,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR22,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR23,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR24,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR25,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR26,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR27,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR28,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR29,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR30,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR31,
        pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR32,
    };
    
    static const pack_RB_fn pack_RB_funcs[] = {
        nullptr,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR1,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR2,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR3,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR4,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR5,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR6,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR7,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR8,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR9,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR10,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR11,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR12,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR13,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR14,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR15,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR16,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR17,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR18,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR19,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR20,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR21,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR22,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR23,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR24,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR25,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR26,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR27,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR28,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR29,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR30,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR31,
        pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR32,
    };

    for (int32_t jc = 0; jc < n; jc += nc) {
        int32_t nc_remain = n - jc;
        int32_t cur_nc = std::min(nc_remain, nc);
        for (int32_t pc = 0; pc < k; pc += kc) {
            int32_t kc_remain = k - pc;
            int32_t cur_kc = std::min(kc_remain, kc);
            int32_t offset_B = jc * ldb + pc;
            float* b_ptr = GET_PTR(B, offset_B);
            int32_t kc_pad = cur_kc;

            kc_pad = ((cur_kc + 3) / 4) * 4;

            int32_t mc_pad = (mc + mr - 1) / mr * mr;
            int32_t nc_pad = (nc + nr - 1) / nr * nr;
            // Zero initialize Ac and Bc so padded elements are strictly zero
            memset(Ac, 0, mc_pad * kc_pad * sizeof(float));
            memset(Bc, 0, kc_pad * nc_pad * sizeof(float));

            pack_CB_funcs[nr](b_ptr, Bc, cur_kc, cur_nc, ldb);
            float beta_actual = GET_BETA(pc, beta);
            for (int32_t ic = 0; ic < m; ic += mc) {
                int32_t mc_remain = m - ic;
                int32_t cur_mc = std::min(mc_remain, mc);
                int32_t offset_A = pc * lda + ic;
                float* a_ptr = GET_PTR(A, offset_A);
                pack_RB_funcs[mr](a_ptr, Ac, cur_mc, cur_kc, lda);
                for (int32_t jr = 0; jr < cur_nc; jr += nr) {
                    int32_t nr_remain = cur_nc - jr;
                    int32_t cur_nr = std::min(nr_remain, nr);
                    for (int32_t ir = 0; ir < cur_mc; ir += mr) {
                        int32_t mr_remain = cur_mc - ir;
                        int32_t cur_mr = std::min(mr_remain, mr);
                        int32_t row_C = ic + ir;
                        int32_t col_C = jc + jr;
                        int32_t offset_C = col_C * ldc + row_C;
                        float* c_ptr = GET_PTR(C, offset_C);
                        float* a_packed = GET_PTR(Ac, ir * kc_pad);
                        float* b_packed = GET_PTR(Bc, jr * kc_pad);
                        gemm_microkernel_v_dispatch_generic(cur_mr, cur_nr, kc_pad, alpha, a_packed, b_packed, beta_actual, c_ptr, ldc, mr, nr);
                    }
                }
            }
        }
    }
}

}
