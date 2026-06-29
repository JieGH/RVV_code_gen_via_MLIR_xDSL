// ============================================================
// xDSL Generated Merged Kernels
// ------------------------------------------------------------
// Precision : fp32
// Types     : A=float, B=float, C=float
// Generation: 2026-06-04 14:28:16
// ============================================================
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

void gemm_RVV_1x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_1x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_1x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_1x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_1x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_1x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_1x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_1x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_3x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_3x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_3x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_3x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_3x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_3x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_3x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_3x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_4x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_4x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_4x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_4x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_4x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_4x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_4x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_4x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_5x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_5x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_5x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_5x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_5x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_5x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_5x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_5x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_6x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_6x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_6x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_6x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_6x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_6x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_6x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_6x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_7x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_7x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_7x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_7x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_7x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_7x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_7x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_7x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_8x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_8x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_8x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_8x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_8x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_8x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_8x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_8x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);

#ifdef __cplusplus
}
#endif
