#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_1x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_1x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_1x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_1x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_2x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_2x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_2x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_2x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);

#ifdef __cplusplus
}
#endif
