// ============================================================
// xDSL Generated Merged Kernels
// ------------------------------------------------------------
// Precision : i8i8i32
// Types     : A=int8_t, B=int8_t, C=int32_t
// Generation: 2026-07-17 04:33:02
// ============================================================
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

void gemm_RVV_1x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x5_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x5_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x6_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x6_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x7_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x7_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x8_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_1x8_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x5_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x5_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x6_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x6_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x7_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x7_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x8_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x8_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x5_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x5_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x6_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x6_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x7_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x7_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x8_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x8_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x5_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x5_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x6_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x6_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x7_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x7_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x8_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x8_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x5_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x5_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x6_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x6_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x7_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x7_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x8_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_5x8_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x5_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x5_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x6_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x6_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x7_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x7_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x8_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_6x8_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x5_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x5_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x6_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x6_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x7_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x7_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x8_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_7x8_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x5_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x5_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x6_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x6_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x7_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x7_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x8_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_8x8_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);

#ifdef __cplusplus
}
#endif
