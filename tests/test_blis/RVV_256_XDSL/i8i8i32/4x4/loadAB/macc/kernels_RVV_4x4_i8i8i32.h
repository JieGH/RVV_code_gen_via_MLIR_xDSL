// ============================================================
// xDSL Generated Merged Kernels
// ------------------------------------------------------------
// Precision : i8i8i32
// Types     : A=int8_t, B=int8_t, C=int32_t
// Generation: 2026-04-27 18:48:41
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
void gemm_RVV_2x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_2x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_3x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x3_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x3_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x4_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);
void gemm_RVV_4x4_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);

#ifdef __cplusplus
}
#endif
