// ============================================================
// xDSL Generated Merged Kernels
// ------------------------------------------------------------
//   Generated   : 2026-03-25 09:33:28
//   Precision   : i8i8i32
//   VLEN_BITS   : 256
// ============================================================
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

void gemm_RVV_1x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_1x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_1x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_1x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x1_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x1_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x2_b0_col_i8i8i32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
void gemm_RVV_2x2_b1_col_i8i8i32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);

#ifdef __cplusplus
}
#endif
