
#pragma once
#ifndef KERNELS_RVV_8X34_FP16_H
#define KERNELS_RVV_8X34_FP16_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


struct exo_win_1f16{
    _Float16 * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f16c{
    const _Float16 * const data;
    const int_fast32_t strides[1];
};
struct exo_win_2f16{
    _Float16 * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f16c{
    const _Float16 * const data;
    const int_fast32_t strides[2];
};
// gemm_RVV_1x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 1] @DRAM
// )
void gemm_RVV_1x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 1] @DRAM
// )
void gemm_RVV_1x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 1] @DRAM
// )
void gemm_RVV_1x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 1] @DRAM
// )
void gemm_RVV_1x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 1] @DRAM
// )
void gemm_RVV_1x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 1] @DRAM
// )
void gemm_RVV_1x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 1] @DRAM
// )
void gemm_RVV_1x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 1] @DRAM
// )
void gemm_RVV_1x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 1] @DRAM
// )
void gemm_RVV_1x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 1] @DRAM
// )
void gemm_RVV_1x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 1] @DRAM
// )
void gemm_RVV_1x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 1] @DRAM
// )
void gemm_RVV_1x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 1] @DRAM
// )
void gemm_RVV_1x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 1] @DRAM
// )
void gemm_RVV_1x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 1] @DRAM
// )
void gemm_RVV_1x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 1] @DRAM
// )
void gemm_RVV_1x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 1] @DRAM
// )
void gemm_RVV_1x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 1] @DRAM
// )
void gemm_RVV_1x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 1] @DRAM
// )
void gemm_RVV_1x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 1] @DRAM
// )
void gemm_RVV_1x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 1] @DRAM
// )
void gemm_RVV_1x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 1] @DRAM
// )
void gemm_RVV_1x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 1] @DRAM
// )
void gemm_RVV_1x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 1] @DRAM
// )
void gemm_RVV_1x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 1] @DRAM
// )
void gemm_RVV_1x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 1] @DRAM
// )
void gemm_RVV_1x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 1] @DRAM
// )
void gemm_RVV_1x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 1] @DRAM
// )
void gemm_RVV_1x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 1] @DRAM
// )
void gemm_RVV_1x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 1] @DRAM
// )
void gemm_RVV_1x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x25_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 1] @DRAM
// )
void gemm_RVV_1x25_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x25_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 1] @DRAM
// )
void gemm_RVV_1x25_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x26_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 1] @DRAM
// )
void gemm_RVV_1x26_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x26_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 1] @DRAM
// )
void gemm_RVV_1x26_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x27_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 1] @DRAM
// )
void gemm_RVV_1x27_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x27_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 1] @DRAM
// )
void gemm_RVV_1x27_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x28_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 1] @DRAM
// )
void gemm_RVV_1x28_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x28_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 1] @DRAM
// )
void gemm_RVV_1x28_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x29_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 1] @DRAM
// )
void gemm_RVV_1x29_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x29_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 1] @DRAM
// )
void gemm_RVV_1x29_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x30_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 1] @DRAM
// )
void gemm_RVV_1x30_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x30_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 1] @DRAM
// )
void gemm_RVV_1x30_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x31_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 1] @DRAM
// )
void gemm_RVV_1x31_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x31_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 1] @DRAM
// )
void gemm_RVV_1x31_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x32_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 1] @DRAM
// )
void gemm_RVV_1x32_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x32_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 1] @DRAM
// )
void gemm_RVV_1x32_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x33_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 1] @DRAM
// )
void gemm_RVV_1x33_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x33_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 1] @DRAM
// )
void gemm_RVV_1x33_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x34_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 1] @DRAM
// )
void gemm_RVV_1x34_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x34_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 1] @DRAM
// )
void gemm_RVV_1x34_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RVV_1x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RVV_1x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RVV_1x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RVV_1x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RVV_1x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RVV_1x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RVV_1x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RVV_1x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RVV_1x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RVV_1x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RVV_1x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RVV_1x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 1] @DRAM
// )
void gemm_RVV_1x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_1x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 1] @DRAM
// )
void gemm_RVV_1x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 2] @DRAM
// )
void gemm_RVV_2x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 2] @DRAM
// )
void gemm_RVV_2x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 2] @DRAM
// )
void gemm_RVV_2x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 2] @DRAM
// )
void gemm_RVV_2x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 2] @DRAM
// )
void gemm_RVV_2x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 2] @DRAM
// )
void gemm_RVV_2x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 2] @DRAM
// )
void gemm_RVV_2x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 2] @DRAM
// )
void gemm_RVV_2x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 2] @DRAM
// )
void gemm_RVV_2x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 2] @DRAM
// )
void gemm_RVV_2x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 2] @DRAM
// )
void gemm_RVV_2x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 2] @DRAM
// )
void gemm_RVV_2x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 2] @DRAM
// )
void gemm_RVV_2x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 2] @DRAM
// )
void gemm_RVV_2x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 2] @DRAM
// )
void gemm_RVV_2x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 2] @DRAM
// )
void gemm_RVV_2x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 2] @DRAM
// )
void gemm_RVV_2x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 2] @DRAM
// )
void gemm_RVV_2x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 2] @DRAM
// )
void gemm_RVV_2x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 2] @DRAM
// )
void gemm_RVV_2x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 2] @DRAM
// )
void gemm_RVV_2x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 2] @DRAM
// )
void gemm_RVV_2x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 2] @DRAM
// )
void gemm_RVV_2x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 2] @DRAM
// )
void gemm_RVV_2x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 2] @DRAM
// )
void gemm_RVV_2x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 2] @DRAM
// )
void gemm_RVV_2x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 2] @DRAM
// )
void gemm_RVV_2x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 2] @DRAM
// )
void gemm_RVV_2x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 2] @DRAM
// )
void gemm_RVV_2x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 2] @DRAM
// )
void gemm_RVV_2x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x25_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 2] @DRAM
// )
void gemm_RVV_2x25_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x25_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 2] @DRAM
// )
void gemm_RVV_2x25_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x26_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 2] @DRAM
// )
void gemm_RVV_2x26_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x26_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 2] @DRAM
// )
void gemm_RVV_2x26_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x27_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 2] @DRAM
// )
void gemm_RVV_2x27_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x27_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 2] @DRAM
// )
void gemm_RVV_2x27_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x28_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 2] @DRAM
// )
void gemm_RVV_2x28_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x28_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 2] @DRAM
// )
void gemm_RVV_2x28_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x29_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 2] @DRAM
// )
void gemm_RVV_2x29_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x29_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 2] @DRAM
// )
void gemm_RVV_2x29_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x30_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 2] @DRAM
// )
void gemm_RVV_2x30_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x30_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 2] @DRAM
// )
void gemm_RVV_2x30_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x31_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 2] @DRAM
// )
void gemm_RVV_2x31_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x31_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 2] @DRAM
// )
void gemm_RVV_2x31_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x32_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 2] @DRAM
// )
void gemm_RVV_2x32_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x32_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 2] @DRAM
// )
void gemm_RVV_2x32_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x33_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 2] @DRAM
// )
void gemm_RVV_2x33_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x33_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 2] @DRAM
// )
void gemm_RVV_2x33_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x34_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 2] @DRAM
// )
void gemm_RVV_2x34_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x34_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 2] @DRAM
// )
void gemm_RVV_2x34_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RVV_2x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RVV_2x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RVV_2x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RVV_2x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RVV_2x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RVV_2x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RVV_2x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RVV_2x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RVV_2x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RVV_2x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RVV_2x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RVV_2x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 2] @DRAM
// )
void gemm_RVV_2x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_2x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 2] @DRAM
// )
void gemm_RVV_2x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 3] @DRAM
// )
void gemm_RVV_3x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 3] @DRAM
// )
void gemm_RVV_3x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 3] @DRAM
// )
void gemm_RVV_3x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 3] @DRAM
// )
void gemm_RVV_3x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 3] @DRAM
// )
void gemm_RVV_3x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 3] @DRAM
// )
void gemm_RVV_3x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 3] @DRAM
// )
void gemm_RVV_3x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 3] @DRAM
// )
void gemm_RVV_3x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 3] @DRAM
// )
void gemm_RVV_3x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 3] @DRAM
// )
void gemm_RVV_3x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 3] @DRAM
// )
void gemm_RVV_3x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 3] @DRAM
// )
void gemm_RVV_3x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 3] @DRAM
// )
void gemm_RVV_3x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 3] @DRAM
// )
void gemm_RVV_3x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 3] @DRAM
// )
void gemm_RVV_3x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 3] @DRAM
// )
void gemm_RVV_3x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 3] @DRAM
// )
void gemm_RVV_3x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 3] @DRAM
// )
void gemm_RVV_3x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 3] @DRAM
// )
void gemm_RVV_3x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 3] @DRAM
// )
void gemm_RVV_3x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 3] @DRAM
// )
void gemm_RVV_3x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 3] @DRAM
// )
void gemm_RVV_3x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 3] @DRAM
// )
void gemm_RVV_3x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 3] @DRAM
// )
void gemm_RVV_3x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 3] @DRAM
// )
void gemm_RVV_3x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 3] @DRAM
// )
void gemm_RVV_3x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 3] @DRAM
// )
void gemm_RVV_3x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 3] @DRAM
// )
void gemm_RVV_3x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 3] @DRAM
// )
void gemm_RVV_3x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 3] @DRAM
// )
void gemm_RVV_3x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x25_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 3] @DRAM
// )
void gemm_RVV_3x25_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x25_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 3] @DRAM
// )
void gemm_RVV_3x25_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x26_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 3] @DRAM
// )
void gemm_RVV_3x26_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x26_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 3] @DRAM
// )
void gemm_RVV_3x26_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x27_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 3] @DRAM
// )
void gemm_RVV_3x27_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x27_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 3] @DRAM
// )
void gemm_RVV_3x27_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x28_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 3] @DRAM
// )
void gemm_RVV_3x28_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x28_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 3] @DRAM
// )
void gemm_RVV_3x28_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x29_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 3] @DRAM
// )
void gemm_RVV_3x29_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x29_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 3] @DRAM
// )
void gemm_RVV_3x29_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x30_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 3] @DRAM
// )
void gemm_RVV_3x30_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x30_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 3] @DRAM
// )
void gemm_RVV_3x30_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x31_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 3] @DRAM
// )
void gemm_RVV_3x31_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x31_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 3] @DRAM
// )
void gemm_RVV_3x31_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x32_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 3] @DRAM
// )
void gemm_RVV_3x32_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x32_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 3] @DRAM
// )
void gemm_RVV_3x32_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x33_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 3] @DRAM
// )
void gemm_RVV_3x33_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x33_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 3] @DRAM
// )
void gemm_RVV_3x33_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x34_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 3] @DRAM
// )
void gemm_RVV_3x34_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x34_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 3] @DRAM
// )
void gemm_RVV_3x34_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RVV_3x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RVV_3x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RVV_3x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RVV_3x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RVV_3x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RVV_3x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RVV_3x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RVV_3x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RVV_3x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RVV_3x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RVV_3x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RVV_3x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 3] @DRAM
// )
void gemm_RVV_3x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_3x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 3] @DRAM
// )
void gemm_RVV_3x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 4] @DRAM
// )
void gemm_RVV_4x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 4] @DRAM
// )
void gemm_RVV_4x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 4] @DRAM
// )
void gemm_RVV_4x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 4] @DRAM
// )
void gemm_RVV_4x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 4] @DRAM
// )
void gemm_RVV_4x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 4] @DRAM
// )
void gemm_RVV_4x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 4] @DRAM
// )
void gemm_RVV_4x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 4] @DRAM
// )
void gemm_RVV_4x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 4] @DRAM
// )
void gemm_RVV_4x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 4] @DRAM
// )
void gemm_RVV_4x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 4] @DRAM
// )
void gemm_RVV_4x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 4] @DRAM
// )
void gemm_RVV_4x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 4] @DRAM
// )
void gemm_RVV_4x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 4] @DRAM
// )
void gemm_RVV_4x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 4] @DRAM
// )
void gemm_RVV_4x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 4] @DRAM
// )
void gemm_RVV_4x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 4] @DRAM
// )
void gemm_RVV_4x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 4] @DRAM
// )
void gemm_RVV_4x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 4] @DRAM
// )
void gemm_RVV_4x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 4] @DRAM
// )
void gemm_RVV_4x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 4] @DRAM
// )
void gemm_RVV_4x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 4] @DRAM
// )
void gemm_RVV_4x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 4] @DRAM
// )
void gemm_RVV_4x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 4] @DRAM
// )
void gemm_RVV_4x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 4] @DRAM
// )
void gemm_RVV_4x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 4] @DRAM
// )
void gemm_RVV_4x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 4] @DRAM
// )
void gemm_RVV_4x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 4] @DRAM
// )
void gemm_RVV_4x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 4] @DRAM
// )
void gemm_RVV_4x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 4] @DRAM
// )
void gemm_RVV_4x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x25_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 4] @DRAM
// )
void gemm_RVV_4x25_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x25_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 4] @DRAM
// )
void gemm_RVV_4x25_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x26_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 4] @DRAM
// )
void gemm_RVV_4x26_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x26_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 4] @DRAM
// )
void gemm_RVV_4x26_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x27_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 4] @DRAM
// )
void gemm_RVV_4x27_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x27_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 4] @DRAM
// )
void gemm_RVV_4x27_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x28_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 4] @DRAM
// )
void gemm_RVV_4x28_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x28_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 4] @DRAM
// )
void gemm_RVV_4x28_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x29_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 4] @DRAM
// )
void gemm_RVV_4x29_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x29_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 4] @DRAM
// )
void gemm_RVV_4x29_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x30_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 4] @DRAM
// )
void gemm_RVV_4x30_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x30_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 4] @DRAM
// )
void gemm_RVV_4x30_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x31_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 4] @DRAM
// )
void gemm_RVV_4x31_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x31_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 4] @DRAM
// )
void gemm_RVV_4x31_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x32_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 4] @DRAM
// )
void gemm_RVV_4x32_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x32_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 4] @DRAM
// )
void gemm_RVV_4x32_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x33_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 4] @DRAM
// )
void gemm_RVV_4x33_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x33_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 4] @DRAM
// )
void gemm_RVV_4x33_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x34_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 4] @DRAM
// )
void gemm_RVV_4x34_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x34_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 4] @DRAM
// )
void gemm_RVV_4x34_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RVV_4x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RVV_4x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RVV_4x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RVV_4x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RVV_4x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RVV_4x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RVV_4x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RVV_4x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RVV_4x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RVV_4x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RVV_4x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RVV_4x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 4] @DRAM
// )
void gemm_RVV_4x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_4x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 4] @DRAM
// )
void gemm_RVV_4x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 5] @DRAM
// )
void gemm_RVV_5x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 5] @DRAM
// )
void gemm_RVV_5x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 5] @DRAM
// )
void gemm_RVV_5x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 5] @DRAM
// )
void gemm_RVV_5x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 5] @DRAM
// )
void gemm_RVV_5x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 5] @DRAM
// )
void gemm_RVV_5x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 5] @DRAM
// )
void gemm_RVV_5x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 5] @DRAM
// )
void gemm_RVV_5x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 5] @DRAM
// )
void gemm_RVV_5x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 5] @DRAM
// )
void gemm_RVV_5x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 5] @DRAM
// )
void gemm_RVV_5x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 5] @DRAM
// )
void gemm_RVV_5x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 5] @DRAM
// )
void gemm_RVV_5x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 5] @DRAM
// )
void gemm_RVV_5x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 5] @DRAM
// )
void gemm_RVV_5x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 5] @DRAM
// )
void gemm_RVV_5x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 5] @DRAM
// )
void gemm_RVV_5x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 5] @DRAM
// )
void gemm_RVV_5x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 5] @DRAM
// )
void gemm_RVV_5x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 5] @DRAM
// )
void gemm_RVV_5x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 5] @DRAM
// )
void gemm_RVV_5x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 5] @DRAM
// )
void gemm_RVV_5x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 5] @DRAM
// )
void gemm_RVV_5x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 5] @DRAM
// )
void gemm_RVV_5x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 5] @DRAM
// )
void gemm_RVV_5x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 5] @DRAM
// )
void gemm_RVV_5x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 5] @DRAM
// )
void gemm_RVV_5x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 5] @DRAM
// )
void gemm_RVV_5x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 5] @DRAM
// )
void gemm_RVV_5x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 5] @DRAM
// )
void gemm_RVV_5x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x25_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 5] @DRAM
// )
void gemm_RVV_5x25_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x25_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 5] @DRAM
// )
void gemm_RVV_5x25_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x26_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 5] @DRAM
// )
void gemm_RVV_5x26_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x26_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 5] @DRAM
// )
void gemm_RVV_5x26_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x27_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 5] @DRAM
// )
void gemm_RVV_5x27_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x27_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 5] @DRAM
// )
void gemm_RVV_5x27_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x28_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 5] @DRAM
// )
void gemm_RVV_5x28_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x28_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 5] @DRAM
// )
void gemm_RVV_5x28_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x29_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 5] @DRAM
// )
void gemm_RVV_5x29_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x29_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 5] @DRAM
// )
void gemm_RVV_5x29_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x30_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 5] @DRAM
// )
void gemm_RVV_5x30_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x30_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 5] @DRAM
// )
void gemm_RVV_5x30_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x31_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 5] @DRAM
// )
void gemm_RVV_5x31_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x31_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 5] @DRAM
// )
void gemm_RVV_5x31_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x32_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 5] @DRAM
// )
void gemm_RVV_5x32_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x32_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 5] @DRAM
// )
void gemm_RVV_5x32_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x33_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 5] @DRAM
// )
void gemm_RVV_5x33_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x33_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 5] @DRAM
// )
void gemm_RVV_5x33_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x34_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 5] @DRAM
// )
void gemm_RVV_5x34_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x34_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 5] @DRAM
// )
void gemm_RVV_5x34_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RVV_5x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RVV_5x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RVV_5x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RVV_5x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RVV_5x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RVV_5x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RVV_5x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RVV_5x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RVV_5x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RVV_5x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RVV_5x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RVV_5x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 5] @DRAM
// )
void gemm_RVV_5x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_5x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 5] @DRAM
// )
void gemm_RVV_5x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 6] @DRAM
// )
void gemm_RVV_6x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 6] @DRAM
// )
void gemm_RVV_6x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 6] @DRAM
// )
void gemm_RVV_6x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 6] @DRAM
// )
void gemm_RVV_6x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 6] @DRAM
// )
void gemm_RVV_6x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 6] @DRAM
// )
void gemm_RVV_6x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 6] @DRAM
// )
void gemm_RVV_6x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 6] @DRAM
// )
void gemm_RVV_6x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 6] @DRAM
// )
void gemm_RVV_6x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 6] @DRAM
// )
void gemm_RVV_6x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 6] @DRAM
// )
void gemm_RVV_6x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 6] @DRAM
// )
void gemm_RVV_6x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 6] @DRAM
// )
void gemm_RVV_6x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 6] @DRAM
// )
void gemm_RVV_6x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 6] @DRAM
// )
void gemm_RVV_6x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 6] @DRAM
// )
void gemm_RVV_6x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 6] @DRAM
// )
void gemm_RVV_6x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 6] @DRAM
// )
void gemm_RVV_6x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 6] @DRAM
// )
void gemm_RVV_6x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 6] @DRAM
// )
void gemm_RVV_6x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 6] @DRAM
// )
void gemm_RVV_6x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 6] @DRAM
// )
void gemm_RVV_6x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 6] @DRAM
// )
void gemm_RVV_6x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 6] @DRAM
// )
void gemm_RVV_6x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 6] @DRAM
// )
void gemm_RVV_6x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 6] @DRAM
// )
void gemm_RVV_6x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 6] @DRAM
// )
void gemm_RVV_6x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 6] @DRAM
// )
void gemm_RVV_6x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 6] @DRAM
// )
void gemm_RVV_6x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 6] @DRAM
// )
void gemm_RVV_6x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x25_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 6] @DRAM
// )
void gemm_RVV_6x25_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x25_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 6] @DRAM
// )
void gemm_RVV_6x25_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x26_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 6] @DRAM
// )
void gemm_RVV_6x26_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x26_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 6] @DRAM
// )
void gemm_RVV_6x26_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x27_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 6] @DRAM
// )
void gemm_RVV_6x27_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x27_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 6] @DRAM
// )
void gemm_RVV_6x27_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x28_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 6] @DRAM
// )
void gemm_RVV_6x28_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x28_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 6] @DRAM
// )
void gemm_RVV_6x28_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x29_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 6] @DRAM
// )
void gemm_RVV_6x29_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x29_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 6] @DRAM
// )
void gemm_RVV_6x29_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x30_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 6] @DRAM
// )
void gemm_RVV_6x30_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x30_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 6] @DRAM
// )
void gemm_RVV_6x30_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x31_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 6] @DRAM
// )
void gemm_RVV_6x31_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x31_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 6] @DRAM
// )
void gemm_RVV_6x31_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x32_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 6] @DRAM
// )
void gemm_RVV_6x32_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x32_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 6] @DRAM
// )
void gemm_RVV_6x32_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x33_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 6] @DRAM
// )
void gemm_RVV_6x33_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x33_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 6] @DRAM
// )
void gemm_RVV_6x33_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x34_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 6] @DRAM
// )
void gemm_RVV_6x34_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x34_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 6] @DRAM
// )
void gemm_RVV_6x34_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RVV_6x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RVV_6x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RVV_6x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RVV_6x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RVV_6x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RVV_6x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RVV_6x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RVV_6x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RVV_6x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RVV_6x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RVV_6x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RVV_6x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 6] @DRAM
// )
void gemm_RVV_6x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_6x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 6] @DRAM
// )
void gemm_RVV_6x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 7] @DRAM
// )
void gemm_RVV_7x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 7] @DRAM
// )
void gemm_RVV_7x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 7] @DRAM
// )
void gemm_RVV_7x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 7] @DRAM
// )
void gemm_RVV_7x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 7] @DRAM
// )
void gemm_RVV_7x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 7] @DRAM
// )
void gemm_RVV_7x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 7] @DRAM
// )
void gemm_RVV_7x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 7] @DRAM
// )
void gemm_RVV_7x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 7] @DRAM
// )
void gemm_RVV_7x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 7] @DRAM
// )
void gemm_RVV_7x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 7] @DRAM
// )
void gemm_RVV_7x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 7] @DRAM
// )
void gemm_RVV_7x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 7] @DRAM
// )
void gemm_RVV_7x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 7] @DRAM
// )
void gemm_RVV_7x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 7] @DRAM
// )
void gemm_RVV_7x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 7] @DRAM
// )
void gemm_RVV_7x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 7] @DRAM
// )
void gemm_RVV_7x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 7] @DRAM
// )
void gemm_RVV_7x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 7] @DRAM
// )
void gemm_RVV_7x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 7] @DRAM
// )
void gemm_RVV_7x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 7] @DRAM
// )
void gemm_RVV_7x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 7] @DRAM
// )
void gemm_RVV_7x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 7] @DRAM
// )
void gemm_RVV_7x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 7] @DRAM
// )
void gemm_RVV_7x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 7] @DRAM
// )
void gemm_RVV_7x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 7] @DRAM
// )
void gemm_RVV_7x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 7] @DRAM
// )
void gemm_RVV_7x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 7] @DRAM
// )
void gemm_RVV_7x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 7] @DRAM
// )
void gemm_RVV_7x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 7] @DRAM
// )
void gemm_RVV_7x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x25_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 7] @DRAM
// )
void gemm_RVV_7x25_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x25_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 7] @DRAM
// )
void gemm_RVV_7x25_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x26_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 7] @DRAM
// )
void gemm_RVV_7x26_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x26_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 7] @DRAM
// )
void gemm_RVV_7x26_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x27_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 7] @DRAM
// )
void gemm_RVV_7x27_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x27_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 7] @DRAM
// )
void gemm_RVV_7x27_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x28_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 7] @DRAM
// )
void gemm_RVV_7x28_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x28_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 7] @DRAM
// )
void gemm_RVV_7x28_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x29_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 7] @DRAM
// )
void gemm_RVV_7x29_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x29_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 7] @DRAM
// )
void gemm_RVV_7x29_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x30_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 7] @DRAM
// )
void gemm_RVV_7x30_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x30_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 7] @DRAM
// )
void gemm_RVV_7x30_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x31_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 7] @DRAM
// )
void gemm_RVV_7x31_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x31_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 7] @DRAM
// )
void gemm_RVV_7x31_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x32_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 7] @DRAM
// )
void gemm_RVV_7x32_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x32_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 7] @DRAM
// )
void gemm_RVV_7x32_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x33_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 7] @DRAM
// )
void gemm_RVV_7x33_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x33_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 7] @DRAM
// )
void gemm_RVV_7x33_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x34_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 7] @DRAM
// )
void gemm_RVV_7x34_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x34_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 7] @DRAM
// )
void gemm_RVV_7x34_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RVV_7x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RVV_7x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RVV_7x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RVV_7x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RVV_7x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RVV_7x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RVV_7x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RVV_7x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RVV_7x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RVV_7x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RVV_7x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RVV_7x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 7] @DRAM
// )
void gemm_RVV_7x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_7x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 7] @DRAM
// )
void gemm_RVV_7x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 8] @DRAM
// )
void gemm_RVV_8x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 8] @DRAM
// )
void gemm_RVV_8x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 8] @DRAM
// )
void gemm_RVV_8x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 8] @DRAM
// )
void gemm_RVV_8x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 8] @DRAM
// )
void gemm_RVV_8x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 8] @DRAM
// )
void gemm_RVV_8x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 8] @DRAM
// )
void gemm_RVV_8x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 8] @DRAM
// )
void gemm_RVV_8x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 8] @DRAM
// )
void gemm_RVV_8x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 8] @DRAM
// )
void gemm_RVV_8x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 8] @DRAM
// )
void gemm_RVV_8x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 8] @DRAM
// )
void gemm_RVV_8x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 8] @DRAM
// )
void gemm_RVV_8x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 8] @DRAM
// )
void gemm_RVV_8x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 8] @DRAM
// )
void gemm_RVV_8x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 8] @DRAM
// )
void gemm_RVV_8x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 8] @DRAM
// )
void gemm_RVV_8x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 8] @DRAM
// )
void gemm_RVV_8x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 8] @DRAM
// )
void gemm_RVV_8x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 8] @DRAM
// )
void gemm_RVV_8x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 8] @DRAM
// )
void gemm_RVV_8x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 8] @DRAM
// )
void gemm_RVV_8x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 8] @DRAM
// )
void gemm_RVV_8x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 8] @DRAM
// )
void gemm_RVV_8x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 8] @DRAM
// )
void gemm_RVV_8x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 8] @DRAM
// )
void gemm_RVV_8x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 8] @DRAM
// )
void gemm_RVV_8x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 8] @DRAM
// )
void gemm_RVV_8x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 8] @DRAM
// )
void gemm_RVV_8x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 8] @DRAM
// )
void gemm_RVV_8x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x25_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 8] @DRAM
// )
void gemm_RVV_8x25_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x25_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 25] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][25, 8] @DRAM
// )
void gemm_RVV_8x25_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x26_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 8] @DRAM
// )
void gemm_RVV_8x26_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x26_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 26] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][26, 8] @DRAM
// )
void gemm_RVV_8x26_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x27_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 8] @DRAM
// )
void gemm_RVV_8x27_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x27_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 27] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][27, 8] @DRAM
// )
void gemm_RVV_8x27_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x28_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 8] @DRAM
// )
void gemm_RVV_8x28_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x28_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 28] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][28, 8] @DRAM
// )
void gemm_RVV_8x28_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x29_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 8] @DRAM
// )
void gemm_RVV_8x29_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x29_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 29] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][29, 8] @DRAM
// )
void gemm_RVV_8x29_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x30_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 8] @DRAM
// )
void gemm_RVV_8x30_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x30_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 30] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][30, 8] @DRAM
// )
void gemm_RVV_8x30_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x31_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 8] @DRAM
// )
void gemm_RVV_8x31_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x31_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 31] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][31, 8] @DRAM
// )
void gemm_RVV_8x31_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x32_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 8] @DRAM
// )
void gemm_RVV_8x32_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x32_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 32] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][32, 8] @DRAM
// )
void gemm_RVV_8x32_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x33_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 8] @DRAM
// )
void gemm_RVV_8x33_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x33_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 33] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][33, 8] @DRAM
// )
void gemm_RVV_8x33_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x34_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 8] @DRAM
// )
void gemm_RVV_8x34_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x34_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 34] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][34, 8] @DRAM
// )
void gemm_RVV_8x34_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RVV_8x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RVV_8x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RVV_8x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RVV_8x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RVV_8x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RVV_8x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RVV_8x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RVV_8x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RVV_8x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RVV_8x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RVV_8x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RVV_8x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 8] @DRAM
// )
void gemm_RVV_8x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RVV_8x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 8] @DRAM
// )
void gemm_RVV_8x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );



#ifdef __cplusplus
}
#endif
#endif  // KERNELS_RVV_8X34_FP16_H
