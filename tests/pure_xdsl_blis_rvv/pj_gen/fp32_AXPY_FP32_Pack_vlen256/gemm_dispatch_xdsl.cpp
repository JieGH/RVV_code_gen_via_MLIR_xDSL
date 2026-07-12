#include "gemm_headers.h"

extern "C" {

void gemm_microkernel_v_dispatch_fp32_MR8_NR8(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 8) { gemm_microkernel_v_fp32_4x8_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 8) { gemm_microkernel_v_fp32_5x8_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 8) { gemm_microkernel_v_fp32_6x8_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 7) { gemm_microkernel_v_fp32_7x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 8) { gemm_microkernel_v_fp32_7x8_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 4) { gemm_microkernel_v_fp32_8x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 5) { gemm_microkernel_v_fp32_8x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 6) { gemm_microkernel_v_fp32_8x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 7) { gemm_microkernel_v_fp32_8x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 8) { gemm_microkernel_v_fp32_8x8_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=8 NR=8)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 8) { gemm_microkernel_v_fp32_4x8_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 8) { gemm_microkernel_v_fp32_5x8_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 8) { gemm_microkernel_v_fp32_6x8_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 7) { gemm_microkernel_v_fp32_7x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 8) { gemm_microkernel_v_fp32_7x8_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 4) { gemm_microkernel_v_fp32_8x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 5) { gemm_microkernel_v_fp32_8x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 6) { gemm_microkernel_v_fp32_8x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 7) { gemm_microkernel_v_fp32_8x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 8) { gemm_microkernel_v_fp32_8x8_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=8 NR=8)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR8_NR8(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR8_NR8((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR8_NR7(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 7) { gemm_microkernel_v_fp32_7x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 4) { gemm_microkernel_v_fp32_8x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 5) { gemm_microkernel_v_fp32_8x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 6) { gemm_microkernel_v_fp32_8x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 7) { gemm_microkernel_v_fp32_8x7_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=8 NR=7)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 7) { gemm_microkernel_v_fp32_7x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 4) { gemm_microkernel_v_fp32_8x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 5) { gemm_microkernel_v_fp32_8x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 6) { gemm_microkernel_v_fp32_8x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 7) { gemm_microkernel_v_fp32_8x7_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=8 NR=7)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR8_NR7(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR8_NR7((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR8_NR6(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 4) { gemm_microkernel_v_fp32_8x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 5) { gemm_microkernel_v_fp32_8x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 6) { gemm_microkernel_v_fp32_8x6_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=8 NR=6)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 4) { gemm_microkernel_v_fp32_8x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 5) { gemm_microkernel_v_fp32_8x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 6) { gemm_microkernel_v_fp32_8x6_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=8 NR=6)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR8_NR6(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR8_NR6((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR8_NR5(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 4) { gemm_microkernel_v_fp32_8x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 5) { gemm_microkernel_v_fp32_8x5_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=8 NR=5)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 4) { gemm_microkernel_v_fp32_8x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 5) { gemm_microkernel_v_fp32_8x5_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=8 NR=5)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR8_NR5(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR8_NR5((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR8_NR4(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 4) { gemm_microkernel_v_fp32_8x4_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=8 NR=4)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 4) { gemm_microkernel_v_fp32_8x4_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=8 NR=4)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR8_NR4(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR8_NR4((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR8_NR3(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=8 NR=3)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 3) { gemm_microkernel_v_fp32_8x3_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=8 NR=3)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR8_NR3(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR8_NR3((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR8_NR2(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b0((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=8 NR=2)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 1) { gemm_microkernel_v_fp32_8x1_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 8 && nr == 2) { gemm_microkernel_v_fp32_8x2_b1((size_t)kc, &alpha, Ac, (size_t)8, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=8 NR=2)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR8_NR2(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR8_NR2((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR7_NR8(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 8) { gemm_microkernel_v_fp32_4x8_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 8) { gemm_microkernel_v_fp32_5x8_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 8) { gemm_microkernel_v_fp32_6x8_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 7) { gemm_microkernel_v_fp32_7x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 8) { gemm_microkernel_v_fp32_7x8_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=7 NR=8)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 8) { gemm_microkernel_v_fp32_4x8_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 8) { gemm_microkernel_v_fp32_5x8_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 8) { gemm_microkernel_v_fp32_6x8_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 7) { gemm_microkernel_v_fp32_7x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 8) { gemm_microkernel_v_fp32_7x8_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=7 NR=8)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR7_NR8(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR7_NR8((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR7_NR7(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 7) { gemm_microkernel_v_fp32_7x7_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=7 NR=7)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 7) { gemm_microkernel_v_fp32_7x7_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=7 NR=7)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR7_NR7(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR7_NR7((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR7_NR6(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=7 NR=6)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 6) { gemm_microkernel_v_fp32_7x6_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=7 NR=6)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR7_NR6(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR7_NR6((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR7_NR5(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=7 NR=5)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 5) { gemm_microkernel_v_fp32_7x5_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=7 NR=5)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR7_NR5(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR7_NR5((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR7_NR4(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=7 NR=4)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 4) { gemm_microkernel_v_fp32_7x4_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=7 NR=4)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR7_NR4(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR7_NR4((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR7_NR3(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=7 NR=3)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 3) { gemm_microkernel_v_fp32_7x3_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=7 NR=3)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR7_NR3(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR7_NR3((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR7_NR2(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b0((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=7 NR=2)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 1) { gemm_microkernel_v_fp32_7x1_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 7 && nr == 2) { gemm_microkernel_v_fp32_7x2_b1((size_t)kc, &alpha, Ac, (size_t)7, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=7 NR=2)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR7_NR2(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR7_NR2((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR6_NR8(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 8) { gemm_microkernel_v_fp32_4x8_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 8) { gemm_microkernel_v_fp32_5x8_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 8) { gemm_microkernel_v_fp32_6x8_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=6 NR=8)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 8) { gemm_microkernel_v_fp32_4x8_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 8) { gemm_microkernel_v_fp32_5x8_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 8) { gemm_microkernel_v_fp32_6x8_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=6 NR=8)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR6_NR8(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR6_NR8((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR6_NR7(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=6 NR=7)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 7) { gemm_microkernel_v_fp32_6x7_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=6 NR=7)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR6_NR7(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR6_NR7((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR6_NR6(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=6 NR=6)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 6) { gemm_microkernel_v_fp32_6x6_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=6 NR=6)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR6_NR6(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR6_NR6((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR6_NR5(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=6 NR=5)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 5) { gemm_microkernel_v_fp32_6x5_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=6 NR=5)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR6_NR5(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR6_NR5((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR6_NR4(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=6 NR=4)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 4) { gemm_microkernel_v_fp32_6x4_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=6 NR=4)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR6_NR4(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR6_NR4((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR6_NR3(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=6 NR=3)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 3) { gemm_microkernel_v_fp32_6x3_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=6 NR=3)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR6_NR3(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR6_NR3((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR6_NR2(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b0((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=6 NR=2)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 1) { gemm_microkernel_v_fp32_6x1_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 6 && nr == 2) { gemm_microkernel_v_fp32_6x2_b1((size_t)kc, &alpha, Ac, (size_t)6, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=6 NR=2)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR6_NR2(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR6_NR2((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR5_NR8(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 8) { gemm_microkernel_v_fp32_4x8_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 8) { gemm_microkernel_v_fp32_5x8_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=5 NR=8)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 8) { gemm_microkernel_v_fp32_4x8_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 8) { gemm_microkernel_v_fp32_5x8_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=5 NR=8)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR5_NR8(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR5_NR8((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR5_NR7(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=5 NR=7)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 7) { gemm_microkernel_v_fp32_5x7_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=5 NR=7)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR5_NR7(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR5_NR7((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR5_NR6(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=5 NR=6)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 6) { gemm_microkernel_v_fp32_5x6_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=5 NR=6)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR5_NR6(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR5_NR6((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR5_NR5(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=5 NR=5)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 5) { gemm_microkernel_v_fp32_5x5_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=5 NR=5)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR5_NR5(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR5_NR5((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR5_NR4(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=5 NR=4)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 4) { gemm_microkernel_v_fp32_5x4_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=5 NR=4)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR5_NR4(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR5_NR4((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR5_NR3(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=5 NR=3)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 3) { gemm_microkernel_v_fp32_5x3_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=5 NR=3)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR5_NR3(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR5_NR3((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR5_NR2(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b0((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=5 NR=2)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 1) { gemm_microkernel_v_fp32_5x1_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 5 && nr == 2) { gemm_microkernel_v_fp32_5x2_b1((size_t)kc, &alpha, Ac, (size_t)5, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=5 NR=2)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR5_NR2(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR5_NR2((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR4_NR8(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 8) { gemm_microkernel_v_fp32_4x8_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=4 NR=8)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 8) { gemm_microkernel_v_fp32_4x8_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=4 NR=8)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR4_NR8(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR4_NR8((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR4_NR7(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=4 NR=7)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 7) { gemm_microkernel_v_fp32_4x7_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=4 NR=7)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR4_NR7(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR4_NR7((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR4_NR6(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=4 NR=6)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 6) { gemm_microkernel_v_fp32_4x6_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=4 NR=6)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR4_NR6(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR4_NR6((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR4_NR5(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=4 NR=5)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 5) { gemm_microkernel_v_fp32_4x5_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=4 NR=5)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR4_NR5(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR4_NR5((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR4_NR4(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=4 NR=4)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_4x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=4 NR=4)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR4_NR4(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR4_NR4((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR4_NR3(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=4 NR=3)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_4x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=4 NR=3)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR4_NR3(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR4_NR3((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR4_NR2(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=4 NR=2)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_4x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_4x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=4 NR=2)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR4_NR2(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR4_NR2((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR3_NR8(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=3 NR=8)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 8) { gemm_microkernel_v_fp32_3x8_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=3 NR=8)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR3_NR8(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR3_NR8((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR3_NR7(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=3 NR=7)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 7) { gemm_microkernel_v_fp32_3x7_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=3 NR=7)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR3_NR7(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR3_NR7((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR3_NR6(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=3 NR=6)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 6) { gemm_microkernel_v_fp32_3x6_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=3 NR=6)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR3_NR6(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR3_NR6((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR3_NR5(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=3 NR=5)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 5) { gemm_microkernel_v_fp32_3x5_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=3 NR=5)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR3_NR5(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR3_NR5((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR3_NR4(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=3 NR=4)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_3x4_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=3 NR=4)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR3_NR4(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR3_NR4((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR3_NR3(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=3 NR=3)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_3x3_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=3 NR=3)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR3_NR3(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR3_NR3((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR3_NR2(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b0((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=3 NR=2)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_3x1_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_3x2_b1((size_t)kc, &alpha, Ac, (size_t)3, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=3 NR=2)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR3_NR2(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR3_NR2((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR2_NR8(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=2 NR=8)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 8) { gemm_microkernel_v_fp32_1x8_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 8) { gemm_microkernel_v_fp32_2x8_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)8, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=2 NR=8)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR2_NR8(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR2_NR8((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR2_NR7(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=2 NR=7)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 7) { gemm_microkernel_v_fp32_1x7_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 7) { gemm_microkernel_v_fp32_2x7_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)7, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=2 NR=7)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR2_NR7(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR2_NR7((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR2_NR6(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=2 NR=6)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 6) { gemm_microkernel_v_fp32_1x6_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 6) { gemm_microkernel_v_fp32_2x6_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)6, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=2 NR=6)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR2_NR6(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR2_NR6((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR2_NR5(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=2 NR=5)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 5) { gemm_microkernel_v_fp32_1x5_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 5) { gemm_microkernel_v_fp32_2x5_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)5, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=2 NR=5)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR2_NR5(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR2_NR5((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR2_NR4(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=2 NR=4)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_1x4_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_2x4_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=2 NR=4)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR2_NR4(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR2_NR4((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR2_NR3(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=2 NR=3)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_1x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_2x3_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)3, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=2 NR=3)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR2_NR3(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR2_NR3((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

void gemm_microkernel_v_dispatch_fp32_MR2_NR2(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b0((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=2 NR=2)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_1x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_1x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_2x1_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_2x2_b1((size_t)kc, &alpha, Ac, (size_t)2, Bc, (size_t)2, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=2 NR=2)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR2_NR2(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR2_NR2((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}

}
