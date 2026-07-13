#pragma once

#include <stddef.h>
#include <stdint.h>

#define CONFIG_DATATYPE "fp32"
#define CONFIG_METHOD "AXPY_FP32_Pack"
#define CONFIG_VLEN 256
#define CONFIG_ARCH "jbpi2"

#ifdef __cplusplus
extern "C" {
#endif

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR1(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR1(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR1(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR1(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR2(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR2(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR2(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR3(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR3(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR3(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR5(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR5(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR5(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR6(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR6(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR6(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR7(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR7(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR7(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR8(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR8(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR8(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR9(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR9(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR9(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR9(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR10(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR10(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR10(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR10(float*, float*, size_t, size_t, size_t);

void gemm_blis_generic(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc, int mr, int nr);

typedef void (*gemm_blis_fn)(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc, int mr, int nr);

struct KernelVariant {
    const char* name;
    int mr;
    int nr;
    gemm_blis_fn func;
};

static const KernelVariant generated_variants[] = {
    {"MR10_NR10", 10, 10, gemm_blis_generic},
    {"MR10_NR9", 10, 9, gemm_blis_generic},
    {"MR10_NR8", 10, 8, gemm_blis_generic},
    {"MR10_NR7", 10, 7, gemm_blis_generic},
    {"MR10_NR6", 10, 6, gemm_blis_generic},
    {"MR10_NR5", 10, 5, gemm_blis_generic},
    {"MR10_NR4", 10, 4, gemm_blis_generic},
    {"MR10_NR3", 10, 3, gemm_blis_generic},
    {"MR10_NR2", 10, 2, gemm_blis_generic},
    {"MR9_NR10", 9, 10, gemm_blis_generic},
    {"MR9_NR9", 9, 9, gemm_blis_generic},
    {"MR9_NR8", 9, 8, gemm_blis_generic},
    {"MR9_NR7", 9, 7, gemm_blis_generic},
    {"MR9_NR6", 9, 6, gemm_blis_generic},
    {"MR9_NR5", 9, 5, gemm_blis_generic},
    {"MR9_NR4", 9, 4, gemm_blis_generic},
    {"MR9_NR3", 9, 3, gemm_blis_generic},
    {"MR9_NR2", 9, 2, gemm_blis_generic},
    {"MR8_NR10", 8, 10, gemm_blis_generic},
    {"MR8_NR9", 8, 9, gemm_blis_generic},
    {"MR8_NR8", 8, 8, gemm_blis_generic},
    {"MR8_NR7", 8, 7, gemm_blis_generic},
    {"MR8_NR6", 8, 6, gemm_blis_generic},
    {"MR8_NR5", 8, 5, gemm_blis_generic},
    {"MR8_NR4", 8, 4, gemm_blis_generic},
    {"MR8_NR3", 8, 3, gemm_blis_generic},
    {"MR8_NR2", 8, 2, gemm_blis_generic},
    {"MR7_NR10", 7, 10, gemm_blis_generic},
    {"MR7_NR9", 7, 9, gemm_blis_generic},
    {"MR7_NR8", 7, 8, gemm_blis_generic},
    {"MR7_NR7", 7, 7, gemm_blis_generic},
    {"MR7_NR6", 7, 6, gemm_blis_generic},
    {"MR7_NR5", 7, 5, gemm_blis_generic},
    {"MR7_NR4", 7, 4, gemm_blis_generic},
    {"MR7_NR3", 7, 3, gemm_blis_generic},
    {"MR7_NR2", 7, 2, gemm_blis_generic},
    {"MR6_NR10", 6, 10, gemm_blis_generic},
    {"MR6_NR9", 6, 9, gemm_blis_generic},
    {"MR6_NR8", 6, 8, gemm_blis_generic},
    {"MR6_NR7", 6, 7, gemm_blis_generic},
    {"MR6_NR6", 6, 6, gemm_blis_generic},
    {"MR6_NR5", 6, 5, gemm_blis_generic},
    {"MR6_NR4", 6, 4, gemm_blis_generic},
    {"MR6_NR3", 6, 3, gemm_blis_generic},
    {"MR6_NR2", 6, 2, gemm_blis_generic},
    {"MR5_NR10", 5, 10, gemm_blis_generic},
    {"MR5_NR9", 5, 9, gemm_blis_generic},
    {"MR5_NR8", 5, 8, gemm_blis_generic},
    {"MR5_NR7", 5, 7, gemm_blis_generic},
    {"MR5_NR6", 5, 6, gemm_blis_generic},
    {"MR5_NR5", 5, 5, gemm_blis_generic},
    {"MR5_NR4", 5, 4, gemm_blis_generic},
    {"MR5_NR3", 5, 3, gemm_blis_generic},
    {"MR5_NR2", 5, 2, gemm_blis_generic},
    {"MR4_NR10", 4, 10, gemm_blis_generic},
    {"MR4_NR9", 4, 9, gemm_blis_generic},
    {"MR4_NR8", 4, 8, gemm_blis_generic},
    {"MR4_NR7", 4, 7, gemm_blis_generic},
    {"MR4_NR6", 4, 6, gemm_blis_generic},
    {"MR4_NR5", 4, 5, gemm_blis_generic},
    {"MR4_NR4", 4, 4, gemm_blis_generic},
    {"MR4_NR3", 4, 3, gemm_blis_generic},
    {"MR4_NR2", 4, 2, gemm_blis_generic},
    {"MR3_NR10", 3, 10, gemm_blis_generic},
    {"MR3_NR9", 3, 9, gemm_blis_generic},
    {"MR3_NR8", 3, 8, gemm_blis_generic},
    {"MR3_NR7", 3, 7, gemm_blis_generic},
    {"MR3_NR6", 3, 6, gemm_blis_generic},
    {"MR3_NR5", 3, 5, gemm_blis_generic},
    {"MR3_NR4", 3, 4, gemm_blis_generic},
    {"MR3_NR3", 3, 3, gemm_blis_generic},
    {"MR3_NR2", 3, 2, gemm_blis_generic},
    {"MR2_NR10", 2, 10, gemm_blis_generic},
    {"MR2_NR9", 2, 9, gemm_blis_generic},
    {"MR2_NR8", 2, 8, gemm_blis_generic},
    {"MR2_NR7", 2, 7, gemm_blis_generic},
    {"MR2_NR6", 2, 6, gemm_blis_generic},
    {"MR2_NR5", 2, 5, gemm_blis_generic},
    {"MR2_NR4", 2, 4, gemm_blis_generic},
    {"MR2_NR3", 2, 3, gemm_blis_generic},
    {"MR2_NR2", 2, 2, gemm_blis_generic},
};

static const int num_variants = 81;

#ifdef __cplusplus
}
#endif
