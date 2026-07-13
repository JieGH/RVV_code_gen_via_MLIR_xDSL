#include "gemm_headers.h"

extern "C" {

void gemm_microkernel_v_dispatch_generic(size_t cur_mr, size_t cur_nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc, size_t max_mr, size_t max_nr) {
    if (beta == 0.0f) {
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
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for cur_mr=%zu cur_nr=%zu\n", cur_mr, cur_nr); abort();
    }
}

void gemm_blis_generic(int32_t m, int32_t n, int32_t k, float alpha, float* A, int32_t lda, float* B, int32_t ldb, float beta, float* C, int32_t ldc, float* Ac, float* Bc, int32_t mc, int32_t nc, int32_t kc, int32_t mr, int32_t nr) {
    typedef void (*pack_CB_fn)(float*, float*, size_t, size_t, size_t);
    typedef void (*pack_RB_fn)(float*, float*, size_t, size_t, size_t);
    
    static const pack_CB_fn pack_CB_funcs[] = {
        nullptr,
        pack_CB_v_if_fp32_NR1,
        pack_CB_v_if_fp32_NR2,
        pack_CB_v_if_fp32_NR3,
        pack_CB_v_if_fp32_NR4,
        pack_CB_v_if_fp32_NR5,
        pack_CB_v_if_fp32_NR6,
        pack_CB_v_if_fp32_NR7,
        pack_CB_v_if_fp32_NR8,
        pack_CB_v_if_fp32_NR9,
        pack_CB_v_if_fp32_NR10,
        pack_CB_v_if_fp32_NR11,
        pack_CB_v_if_fp32_NR12,
        pack_CB_v_if_fp32_NR13,
        pack_CB_v_if_fp32_NR14,
        pack_CB_v_if_fp32_NR15,
        pack_CB_v_if_fp32_NR16,
    };
    
    static const pack_RB_fn pack_RB_funcs[] = {
        nullptr,
        pack_RB_v_if_fp32_MR1,
        pack_RB_v_if_fp32_MR2,
        pack_RB_v_if_fp32_MR3,
        pack_RB_v_if_fp32_MR4,
        pack_RB_v_if_fp32_MR5,
        pack_RB_v_if_fp32_MR6,
        pack_RB_v_if_fp32_MR7,
        pack_RB_v_if_fp32_MR8,
        pack_RB_v_if_fp32_MR9,
        pack_RB_v_if_fp32_MR10,
        pack_RB_v_if_fp32_MR11,
        pack_RB_v_if_fp32_MR12,
        pack_RB_v_if_fp32_MR13,
        pack_RB_v_if_fp32_MR14,
        pack_RB_v_if_fp32_MR15,
        pack_RB_v_if_fp32_MR16,
    };

    for (int32_t jc = 0; jc < n; jc += nc) {
        int32_t nc_remain = n - jc;
        int32_t cur_nc = std::min(nc_remain, nc);
        for (int32_t pc = 0; pc < k; pc += kc) {
            int32_t kc_remain = k - pc;
            int32_t cur_kc = std::min(kc_remain, kc);
            int32_t offset_B = jc * ldb + pc;
            float* b_ptr = GET_PTR(B, offset_B);
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
                        float* a_packed = GET_PTR(Ac, ir * cur_kc);
                        float* b_packed = GET_PTR(Bc, jr * cur_kc);
                        gemm_microkernel_v_dispatch_generic(cur_mr, cur_nr, cur_kc, alpha, a_packed, b_packed, beta_actual, c_ptr, ldc, mr, nr);
                    }
                }
            }
        }
    }
}

}
