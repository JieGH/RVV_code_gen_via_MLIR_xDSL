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
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR11(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR11(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR11(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR11(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR12(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR12(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR12(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR12(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR13(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR13(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR13(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR13(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR14(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR14(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR14(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR14(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR15(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR15(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR15(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR15(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR16(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR16(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR16(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR16(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR17(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR17(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR17(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR17(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR18(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR18(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR18(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR18(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR19(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR19(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR19(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR19(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR20(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR20(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR20(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR20(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR21(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR21(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR21(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR21(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR22(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR22(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR22(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR22(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR23(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR23(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR23(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR23(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR24(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR24(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR24(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR24(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR25(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR25(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR25(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR25(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR26(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR26(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR26(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR26(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR27(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR27(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR27(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR27(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR28(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR28(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR28(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR28(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR29(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR29(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR29(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR29(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR30(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR30(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR30(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR30(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR31(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR31(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR31(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR31(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR32(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR32(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR32(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR32(float*, float*, size_t, size_t, size_t);

void gemm_blis_generic(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc, int mr, int nr);

typedef void (*gemm_blis_fn)(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc, int mr, int nr);

struct KernelVariant {
    const char* name;
    int mr;
    int nr;
    gemm_blis_fn func;
};

static const KernelVariant generated_variants[] = {
    {"MR32_NR16", 32, 16, gemm_blis_generic},
    {"MR32_NR15", 32, 15, gemm_blis_generic},
    {"MR32_NR14", 32, 14, gemm_blis_generic},
    {"MR32_NR13", 32, 13, gemm_blis_generic},
    {"MR32_NR12", 32, 12, gemm_blis_generic},
    {"MR32_NR11", 32, 11, gemm_blis_generic},
    {"MR32_NR10", 32, 10, gemm_blis_generic},
    {"MR32_NR9", 32, 9, gemm_blis_generic},
    {"MR32_NR8", 32, 8, gemm_blis_generic},
    {"MR32_NR7", 32, 7, gemm_blis_generic},
    {"MR32_NR6", 32, 6, gemm_blis_generic},
    {"MR32_NR5", 32, 5, gemm_blis_generic},
    {"MR32_NR4", 32, 4, gemm_blis_generic},
    {"MR32_NR3", 32, 3, gemm_blis_generic},
    {"MR32_NR2", 32, 2, gemm_blis_generic},
    {"MR31_NR16", 31, 16, gemm_blis_generic},
    {"MR31_NR15", 31, 15, gemm_blis_generic},
    {"MR31_NR14", 31, 14, gemm_blis_generic},
    {"MR31_NR13", 31, 13, gemm_blis_generic},
    {"MR31_NR12", 31, 12, gemm_blis_generic},
    {"MR31_NR11", 31, 11, gemm_blis_generic},
    {"MR31_NR10", 31, 10, gemm_blis_generic},
    {"MR31_NR9", 31, 9, gemm_blis_generic},
    {"MR31_NR8", 31, 8, gemm_blis_generic},
    {"MR31_NR7", 31, 7, gemm_blis_generic},
    {"MR31_NR6", 31, 6, gemm_blis_generic},
    {"MR31_NR5", 31, 5, gemm_blis_generic},
    {"MR31_NR4", 31, 4, gemm_blis_generic},
    {"MR31_NR3", 31, 3, gemm_blis_generic},
    {"MR31_NR2", 31, 2, gemm_blis_generic},
    {"MR30_NR16", 30, 16, gemm_blis_generic},
    {"MR30_NR15", 30, 15, gemm_blis_generic},
    {"MR30_NR14", 30, 14, gemm_blis_generic},
    {"MR30_NR13", 30, 13, gemm_blis_generic},
    {"MR30_NR12", 30, 12, gemm_blis_generic},
    {"MR30_NR11", 30, 11, gemm_blis_generic},
    {"MR30_NR10", 30, 10, gemm_blis_generic},
    {"MR30_NR9", 30, 9, gemm_blis_generic},
    {"MR30_NR8", 30, 8, gemm_blis_generic},
    {"MR30_NR7", 30, 7, gemm_blis_generic},
    {"MR30_NR6", 30, 6, gemm_blis_generic},
    {"MR30_NR5", 30, 5, gemm_blis_generic},
    {"MR30_NR4", 30, 4, gemm_blis_generic},
    {"MR30_NR3", 30, 3, gemm_blis_generic},
    {"MR30_NR2", 30, 2, gemm_blis_generic},
    {"MR29_NR16", 29, 16, gemm_blis_generic},
    {"MR29_NR15", 29, 15, gemm_blis_generic},
    {"MR29_NR14", 29, 14, gemm_blis_generic},
    {"MR29_NR13", 29, 13, gemm_blis_generic},
    {"MR29_NR12", 29, 12, gemm_blis_generic},
    {"MR29_NR11", 29, 11, gemm_blis_generic},
    {"MR29_NR10", 29, 10, gemm_blis_generic},
    {"MR29_NR9", 29, 9, gemm_blis_generic},
    {"MR29_NR8", 29, 8, gemm_blis_generic},
    {"MR29_NR7", 29, 7, gemm_blis_generic},
    {"MR29_NR6", 29, 6, gemm_blis_generic},
    {"MR29_NR5", 29, 5, gemm_blis_generic},
    {"MR29_NR4", 29, 4, gemm_blis_generic},
    {"MR29_NR3", 29, 3, gemm_blis_generic},
    {"MR29_NR2", 29, 2, gemm_blis_generic},
    {"MR28_NR16", 28, 16, gemm_blis_generic},
    {"MR28_NR15", 28, 15, gemm_blis_generic},
    {"MR28_NR14", 28, 14, gemm_blis_generic},
    {"MR28_NR13", 28, 13, gemm_blis_generic},
    {"MR28_NR12", 28, 12, gemm_blis_generic},
    {"MR28_NR11", 28, 11, gemm_blis_generic},
    {"MR28_NR10", 28, 10, gemm_blis_generic},
    {"MR28_NR9", 28, 9, gemm_blis_generic},
    {"MR28_NR8", 28, 8, gemm_blis_generic},
    {"MR28_NR7", 28, 7, gemm_blis_generic},
    {"MR28_NR6", 28, 6, gemm_blis_generic},
    {"MR28_NR5", 28, 5, gemm_blis_generic},
    {"MR28_NR4", 28, 4, gemm_blis_generic},
    {"MR28_NR3", 28, 3, gemm_blis_generic},
    {"MR28_NR2", 28, 2, gemm_blis_generic},
    {"MR27_NR16", 27, 16, gemm_blis_generic},
    {"MR27_NR15", 27, 15, gemm_blis_generic},
    {"MR27_NR14", 27, 14, gemm_blis_generic},
    {"MR27_NR13", 27, 13, gemm_blis_generic},
    {"MR27_NR12", 27, 12, gemm_blis_generic},
    {"MR27_NR11", 27, 11, gemm_blis_generic},
    {"MR27_NR10", 27, 10, gemm_blis_generic},
    {"MR27_NR9", 27, 9, gemm_blis_generic},
    {"MR27_NR8", 27, 8, gemm_blis_generic},
    {"MR27_NR7", 27, 7, gemm_blis_generic},
    {"MR27_NR6", 27, 6, gemm_blis_generic},
    {"MR27_NR5", 27, 5, gemm_blis_generic},
    {"MR27_NR4", 27, 4, gemm_blis_generic},
    {"MR27_NR3", 27, 3, gemm_blis_generic},
    {"MR27_NR2", 27, 2, gemm_blis_generic},
    {"MR26_NR16", 26, 16, gemm_blis_generic},
    {"MR26_NR15", 26, 15, gemm_blis_generic},
    {"MR26_NR14", 26, 14, gemm_blis_generic},
    {"MR26_NR13", 26, 13, gemm_blis_generic},
    {"MR26_NR12", 26, 12, gemm_blis_generic},
    {"MR26_NR11", 26, 11, gemm_blis_generic},
    {"MR26_NR10", 26, 10, gemm_blis_generic},
    {"MR26_NR9", 26, 9, gemm_blis_generic},
    {"MR26_NR8", 26, 8, gemm_blis_generic},
    {"MR26_NR7", 26, 7, gemm_blis_generic},
    {"MR26_NR6", 26, 6, gemm_blis_generic},
    {"MR26_NR5", 26, 5, gemm_blis_generic},
    {"MR26_NR4", 26, 4, gemm_blis_generic},
    {"MR26_NR3", 26, 3, gemm_blis_generic},
    {"MR26_NR2", 26, 2, gemm_blis_generic},
    {"MR25_NR16", 25, 16, gemm_blis_generic},
    {"MR25_NR15", 25, 15, gemm_blis_generic},
    {"MR25_NR14", 25, 14, gemm_blis_generic},
    {"MR25_NR13", 25, 13, gemm_blis_generic},
    {"MR25_NR12", 25, 12, gemm_blis_generic},
    {"MR25_NR11", 25, 11, gemm_blis_generic},
    {"MR25_NR10", 25, 10, gemm_blis_generic},
    {"MR25_NR9", 25, 9, gemm_blis_generic},
    {"MR25_NR8", 25, 8, gemm_blis_generic},
    {"MR25_NR7", 25, 7, gemm_blis_generic},
    {"MR25_NR6", 25, 6, gemm_blis_generic},
    {"MR25_NR5", 25, 5, gemm_blis_generic},
    {"MR25_NR4", 25, 4, gemm_blis_generic},
    {"MR25_NR3", 25, 3, gemm_blis_generic},
    {"MR25_NR2", 25, 2, gemm_blis_generic},
    {"MR24_NR16", 24, 16, gemm_blis_generic},
    {"MR24_NR15", 24, 15, gemm_blis_generic},
    {"MR24_NR14", 24, 14, gemm_blis_generic},
    {"MR24_NR13", 24, 13, gemm_blis_generic},
    {"MR24_NR12", 24, 12, gemm_blis_generic},
    {"MR24_NR11", 24, 11, gemm_blis_generic},
    {"MR24_NR10", 24, 10, gemm_blis_generic},
    {"MR24_NR9", 24, 9, gemm_blis_generic},
    {"MR24_NR8", 24, 8, gemm_blis_generic},
    {"MR24_NR7", 24, 7, gemm_blis_generic},
    {"MR24_NR6", 24, 6, gemm_blis_generic},
    {"MR24_NR5", 24, 5, gemm_blis_generic},
    {"MR24_NR4", 24, 4, gemm_blis_generic},
    {"MR24_NR3", 24, 3, gemm_blis_generic},
    {"MR24_NR2", 24, 2, gemm_blis_generic},
    {"MR23_NR16", 23, 16, gemm_blis_generic},
    {"MR23_NR15", 23, 15, gemm_blis_generic},
    {"MR23_NR14", 23, 14, gemm_blis_generic},
    {"MR23_NR13", 23, 13, gemm_blis_generic},
    {"MR23_NR12", 23, 12, gemm_blis_generic},
    {"MR23_NR11", 23, 11, gemm_blis_generic},
    {"MR23_NR10", 23, 10, gemm_blis_generic},
    {"MR23_NR9", 23, 9, gemm_blis_generic},
    {"MR23_NR8", 23, 8, gemm_blis_generic},
    {"MR23_NR7", 23, 7, gemm_blis_generic},
    {"MR23_NR6", 23, 6, gemm_blis_generic},
    {"MR23_NR5", 23, 5, gemm_blis_generic},
    {"MR23_NR4", 23, 4, gemm_blis_generic},
    {"MR23_NR3", 23, 3, gemm_blis_generic},
    {"MR23_NR2", 23, 2, gemm_blis_generic},
    {"MR22_NR16", 22, 16, gemm_blis_generic},
    {"MR22_NR15", 22, 15, gemm_blis_generic},
    {"MR22_NR14", 22, 14, gemm_blis_generic},
    {"MR22_NR13", 22, 13, gemm_blis_generic},
    {"MR22_NR12", 22, 12, gemm_blis_generic},
    {"MR22_NR11", 22, 11, gemm_blis_generic},
    {"MR22_NR10", 22, 10, gemm_blis_generic},
    {"MR22_NR9", 22, 9, gemm_blis_generic},
    {"MR22_NR8", 22, 8, gemm_blis_generic},
    {"MR22_NR7", 22, 7, gemm_blis_generic},
    {"MR22_NR6", 22, 6, gemm_blis_generic},
    {"MR22_NR5", 22, 5, gemm_blis_generic},
    {"MR22_NR4", 22, 4, gemm_blis_generic},
    {"MR22_NR3", 22, 3, gemm_blis_generic},
    {"MR22_NR2", 22, 2, gemm_blis_generic},
    {"MR21_NR16", 21, 16, gemm_blis_generic},
    {"MR21_NR15", 21, 15, gemm_blis_generic},
    {"MR21_NR14", 21, 14, gemm_blis_generic},
    {"MR21_NR13", 21, 13, gemm_blis_generic},
    {"MR21_NR12", 21, 12, gemm_blis_generic},
    {"MR21_NR11", 21, 11, gemm_blis_generic},
    {"MR21_NR10", 21, 10, gemm_blis_generic},
    {"MR21_NR9", 21, 9, gemm_blis_generic},
    {"MR21_NR8", 21, 8, gemm_blis_generic},
    {"MR21_NR7", 21, 7, gemm_blis_generic},
    {"MR21_NR6", 21, 6, gemm_blis_generic},
    {"MR21_NR5", 21, 5, gemm_blis_generic},
    {"MR21_NR4", 21, 4, gemm_blis_generic},
    {"MR21_NR3", 21, 3, gemm_blis_generic},
    {"MR21_NR2", 21, 2, gemm_blis_generic},
    {"MR20_NR16", 20, 16, gemm_blis_generic},
    {"MR20_NR15", 20, 15, gemm_blis_generic},
    {"MR20_NR14", 20, 14, gemm_blis_generic},
    {"MR20_NR13", 20, 13, gemm_blis_generic},
    {"MR20_NR12", 20, 12, gemm_blis_generic},
    {"MR20_NR11", 20, 11, gemm_blis_generic},
    {"MR20_NR10", 20, 10, gemm_blis_generic},
    {"MR20_NR9", 20, 9, gemm_blis_generic},
    {"MR20_NR8", 20, 8, gemm_blis_generic},
    {"MR20_NR7", 20, 7, gemm_blis_generic},
    {"MR20_NR6", 20, 6, gemm_blis_generic},
    {"MR20_NR5", 20, 5, gemm_blis_generic},
    {"MR20_NR4", 20, 4, gemm_blis_generic},
    {"MR20_NR3", 20, 3, gemm_blis_generic},
    {"MR20_NR2", 20, 2, gemm_blis_generic},
    {"MR19_NR16", 19, 16, gemm_blis_generic},
    {"MR19_NR15", 19, 15, gemm_blis_generic},
    {"MR19_NR14", 19, 14, gemm_blis_generic},
    {"MR19_NR13", 19, 13, gemm_blis_generic},
    {"MR19_NR12", 19, 12, gemm_blis_generic},
    {"MR19_NR11", 19, 11, gemm_blis_generic},
    {"MR19_NR10", 19, 10, gemm_blis_generic},
    {"MR19_NR9", 19, 9, gemm_blis_generic},
    {"MR19_NR8", 19, 8, gemm_blis_generic},
    {"MR19_NR7", 19, 7, gemm_blis_generic},
    {"MR19_NR6", 19, 6, gemm_blis_generic},
    {"MR19_NR5", 19, 5, gemm_blis_generic},
    {"MR19_NR4", 19, 4, gemm_blis_generic},
    {"MR19_NR3", 19, 3, gemm_blis_generic},
    {"MR19_NR2", 19, 2, gemm_blis_generic},
    {"MR18_NR16", 18, 16, gemm_blis_generic},
    {"MR18_NR15", 18, 15, gemm_blis_generic},
    {"MR18_NR14", 18, 14, gemm_blis_generic},
    {"MR18_NR13", 18, 13, gemm_blis_generic},
    {"MR18_NR12", 18, 12, gemm_blis_generic},
    {"MR18_NR11", 18, 11, gemm_blis_generic},
    {"MR18_NR10", 18, 10, gemm_blis_generic},
    {"MR18_NR9", 18, 9, gemm_blis_generic},
    {"MR18_NR8", 18, 8, gemm_blis_generic},
    {"MR18_NR7", 18, 7, gemm_blis_generic},
    {"MR18_NR6", 18, 6, gemm_blis_generic},
    {"MR18_NR5", 18, 5, gemm_blis_generic},
    {"MR18_NR4", 18, 4, gemm_blis_generic},
    {"MR18_NR3", 18, 3, gemm_blis_generic},
    {"MR18_NR2", 18, 2, gemm_blis_generic},
    {"MR17_NR16", 17, 16, gemm_blis_generic},
    {"MR17_NR15", 17, 15, gemm_blis_generic},
    {"MR17_NR14", 17, 14, gemm_blis_generic},
    {"MR17_NR13", 17, 13, gemm_blis_generic},
    {"MR17_NR12", 17, 12, gemm_blis_generic},
    {"MR17_NR11", 17, 11, gemm_blis_generic},
    {"MR17_NR10", 17, 10, gemm_blis_generic},
    {"MR17_NR9", 17, 9, gemm_blis_generic},
    {"MR17_NR8", 17, 8, gemm_blis_generic},
    {"MR17_NR7", 17, 7, gemm_blis_generic},
    {"MR17_NR6", 17, 6, gemm_blis_generic},
    {"MR17_NR5", 17, 5, gemm_blis_generic},
    {"MR17_NR4", 17, 4, gemm_blis_generic},
    {"MR17_NR3", 17, 3, gemm_blis_generic},
    {"MR17_NR2", 17, 2, gemm_blis_generic},
    {"MR16_NR16", 16, 16, gemm_blis_generic},
    {"MR16_NR15", 16, 15, gemm_blis_generic},
    {"MR16_NR14", 16, 14, gemm_blis_generic},
    {"MR16_NR13", 16, 13, gemm_blis_generic},
    {"MR16_NR12", 16, 12, gemm_blis_generic},
    {"MR16_NR11", 16, 11, gemm_blis_generic},
    {"MR16_NR10", 16, 10, gemm_blis_generic},
    {"MR16_NR9", 16, 9, gemm_blis_generic},
    {"MR16_NR8", 16, 8, gemm_blis_generic},
    {"MR16_NR7", 16, 7, gemm_blis_generic},
    {"MR16_NR6", 16, 6, gemm_blis_generic},
    {"MR16_NR5", 16, 5, gemm_blis_generic},
    {"MR16_NR4", 16, 4, gemm_blis_generic},
    {"MR16_NR3", 16, 3, gemm_blis_generic},
    {"MR16_NR2", 16, 2, gemm_blis_generic},
    {"MR15_NR16", 15, 16, gemm_blis_generic},
    {"MR15_NR15", 15, 15, gemm_blis_generic},
    {"MR15_NR14", 15, 14, gemm_blis_generic},
    {"MR15_NR13", 15, 13, gemm_blis_generic},
    {"MR15_NR12", 15, 12, gemm_blis_generic},
    {"MR15_NR11", 15, 11, gemm_blis_generic},
    {"MR15_NR10", 15, 10, gemm_blis_generic},
    {"MR15_NR9", 15, 9, gemm_blis_generic},
    {"MR15_NR8", 15, 8, gemm_blis_generic},
    {"MR15_NR7", 15, 7, gemm_blis_generic},
    {"MR15_NR6", 15, 6, gemm_blis_generic},
    {"MR15_NR5", 15, 5, gemm_blis_generic},
    {"MR15_NR4", 15, 4, gemm_blis_generic},
    {"MR15_NR3", 15, 3, gemm_blis_generic},
    {"MR15_NR2", 15, 2, gemm_blis_generic},
    {"MR14_NR16", 14, 16, gemm_blis_generic},
    {"MR14_NR15", 14, 15, gemm_blis_generic},
    {"MR14_NR14", 14, 14, gemm_blis_generic},
    {"MR14_NR13", 14, 13, gemm_blis_generic},
    {"MR14_NR12", 14, 12, gemm_blis_generic},
    {"MR14_NR11", 14, 11, gemm_blis_generic},
    {"MR14_NR10", 14, 10, gemm_blis_generic},
    {"MR14_NR9", 14, 9, gemm_blis_generic},
    {"MR14_NR8", 14, 8, gemm_blis_generic},
    {"MR14_NR7", 14, 7, gemm_blis_generic},
    {"MR14_NR6", 14, 6, gemm_blis_generic},
    {"MR14_NR5", 14, 5, gemm_blis_generic},
    {"MR14_NR4", 14, 4, gemm_blis_generic},
    {"MR14_NR3", 14, 3, gemm_blis_generic},
    {"MR14_NR2", 14, 2, gemm_blis_generic},
    {"MR13_NR16", 13, 16, gemm_blis_generic},
    {"MR13_NR15", 13, 15, gemm_blis_generic},
    {"MR13_NR14", 13, 14, gemm_blis_generic},
    {"MR13_NR13", 13, 13, gemm_blis_generic},
    {"MR13_NR12", 13, 12, gemm_blis_generic},
    {"MR13_NR11", 13, 11, gemm_blis_generic},
    {"MR13_NR10", 13, 10, gemm_blis_generic},
    {"MR13_NR9", 13, 9, gemm_blis_generic},
    {"MR13_NR8", 13, 8, gemm_blis_generic},
    {"MR13_NR7", 13, 7, gemm_blis_generic},
    {"MR13_NR6", 13, 6, gemm_blis_generic},
    {"MR13_NR5", 13, 5, gemm_blis_generic},
    {"MR13_NR4", 13, 4, gemm_blis_generic},
    {"MR13_NR3", 13, 3, gemm_blis_generic},
    {"MR13_NR2", 13, 2, gemm_blis_generic},
    {"MR12_NR16", 12, 16, gemm_blis_generic},
    {"MR12_NR15", 12, 15, gemm_blis_generic},
    {"MR12_NR14", 12, 14, gemm_blis_generic},
    {"MR12_NR13", 12, 13, gemm_blis_generic},
    {"MR12_NR12", 12, 12, gemm_blis_generic},
    {"MR12_NR11", 12, 11, gemm_blis_generic},
    {"MR12_NR10", 12, 10, gemm_blis_generic},
    {"MR12_NR9", 12, 9, gemm_blis_generic},
    {"MR12_NR8", 12, 8, gemm_blis_generic},
    {"MR12_NR7", 12, 7, gemm_blis_generic},
    {"MR12_NR6", 12, 6, gemm_blis_generic},
    {"MR12_NR5", 12, 5, gemm_blis_generic},
    {"MR12_NR4", 12, 4, gemm_blis_generic},
    {"MR12_NR3", 12, 3, gemm_blis_generic},
    {"MR12_NR2", 12, 2, gemm_blis_generic},
    {"MR11_NR16", 11, 16, gemm_blis_generic},
    {"MR11_NR15", 11, 15, gemm_blis_generic},
    {"MR11_NR14", 11, 14, gemm_blis_generic},
    {"MR11_NR13", 11, 13, gemm_blis_generic},
    {"MR11_NR12", 11, 12, gemm_blis_generic},
    {"MR11_NR11", 11, 11, gemm_blis_generic},
    {"MR11_NR10", 11, 10, gemm_blis_generic},
    {"MR11_NR9", 11, 9, gemm_blis_generic},
    {"MR11_NR8", 11, 8, gemm_blis_generic},
    {"MR11_NR7", 11, 7, gemm_blis_generic},
    {"MR11_NR6", 11, 6, gemm_blis_generic},
    {"MR11_NR5", 11, 5, gemm_blis_generic},
    {"MR11_NR4", 11, 4, gemm_blis_generic},
    {"MR11_NR3", 11, 3, gemm_blis_generic},
    {"MR11_NR2", 11, 2, gemm_blis_generic},
    {"MR10_NR16", 10, 16, gemm_blis_generic},
    {"MR10_NR15", 10, 15, gemm_blis_generic},
    {"MR10_NR14", 10, 14, gemm_blis_generic},
    {"MR10_NR13", 10, 13, gemm_blis_generic},
    {"MR10_NR12", 10, 12, gemm_blis_generic},
    {"MR10_NR11", 10, 11, gemm_blis_generic},
    {"MR10_NR10", 10, 10, gemm_blis_generic},
    {"MR10_NR9", 10, 9, gemm_blis_generic},
    {"MR10_NR8", 10, 8, gemm_blis_generic},
    {"MR10_NR7", 10, 7, gemm_blis_generic},
    {"MR10_NR6", 10, 6, gemm_blis_generic},
    {"MR10_NR5", 10, 5, gemm_blis_generic},
    {"MR10_NR4", 10, 4, gemm_blis_generic},
    {"MR10_NR3", 10, 3, gemm_blis_generic},
    {"MR10_NR2", 10, 2, gemm_blis_generic},
    {"MR9_NR16", 9, 16, gemm_blis_generic},
    {"MR9_NR15", 9, 15, gemm_blis_generic},
    {"MR9_NR14", 9, 14, gemm_blis_generic},
    {"MR9_NR13", 9, 13, gemm_blis_generic},
    {"MR9_NR12", 9, 12, gemm_blis_generic},
    {"MR9_NR11", 9, 11, gemm_blis_generic},
    {"MR9_NR10", 9, 10, gemm_blis_generic},
    {"MR9_NR9", 9, 9, gemm_blis_generic},
    {"MR9_NR8", 9, 8, gemm_blis_generic},
    {"MR9_NR7", 9, 7, gemm_blis_generic},
    {"MR9_NR6", 9, 6, gemm_blis_generic},
    {"MR9_NR5", 9, 5, gemm_blis_generic},
    {"MR9_NR4", 9, 4, gemm_blis_generic},
    {"MR9_NR3", 9, 3, gemm_blis_generic},
    {"MR9_NR2", 9, 2, gemm_blis_generic},
    {"MR8_NR16", 8, 16, gemm_blis_generic},
    {"MR8_NR15", 8, 15, gemm_blis_generic},
    {"MR8_NR14", 8, 14, gemm_blis_generic},
    {"MR8_NR13", 8, 13, gemm_blis_generic},
    {"MR8_NR12", 8, 12, gemm_blis_generic},
    {"MR8_NR11", 8, 11, gemm_blis_generic},
    {"MR8_NR10", 8, 10, gemm_blis_generic},
    {"MR8_NR9", 8, 9, gemm_blis_generic},
    {"MR8_NR8", 8, 8, gemm_blis_generic},
    {"MR8_NR7", 8, 7, gemm_blis_generic},
    {"MR8_NR6", 8, 6, gemm_blis_generic},
    {"MR8_NR5", 8, 5, gemm_blis_generic},
    {"MR8_NR4", 8, 4, gemm_blis_generic},
    {"MR8_NR3", 8, 3, gemm_blis_generic},
    {"MR8_NR2", 8, 2, gemm_blis_generic},
    {"MR7_NR16", 7, 16, gemm_blis_generic},
    {"MR7_NR15", 7, 15, gemm_blis_generic},
    {"MR7_NR14", 7, 14, gemm_blis_generic},
    {"MR7_NR13", 7, 13, gemm_blis_generic},
    {"MR7_NR12", 7, 12, gemm_blis_generic},
    {"MR7_NR11", 7, 11, gemm_blis_generic},
    {"MR7_NR10", 7, 10, gemm_blis_generic},
    {"MR7_NR9", 7, 9, gemm_blis_generic},
    {"MR7_NR8", 7, 8, gemm_blis_generic},
    {"MR7_NR7", 7, 7, gemm_blis_generic},
    {"MR7_NR6", 7, 6, gemm_blis_generic},
    {"MR7_NR5", 7, 5, gemm_blis_generic},
    {"MR7_NR4", 7, 4, gemm_blis_generic},
    {"MR7_NR3", 7, 3, gemm_blis_generic},
    {"MR7_NR2", 7, 2, gemm_blis_generic},
    {"MR6_NR16", 6, 16, gemm_blis_generic},
    {"MR6_NR15", 6, 15, gemm_blis_generic},
    {"MR6_NR14", 6, 14, gemm_blis_generic},
    {"MR6_NR13", 6, 13, gemm_blis_generic},
    {"MR6_NR12", 6, 12, gemm_blis_generic},
    {"MR6_NR11", 6, 11, gemm_blis_generic},
    {"MR6_NR10", 6, 10, gemm_blis_generic},
    {"MR6_NR9", 6, 9, gemm_blis_generic},
    {"MR6_NR8", 6, 8, gemm_blis_generic},
    {"MR6_NR7", 6, 7, gemm_blis_generic},
    {"MR6_NR6", 6, 6, gemm_blis_generic},
    {"MR6_NR5", 6, 5, gemm_blis_generic},
    {"MR6_NR4", 6, 4, gemm_blis_generic},
    {"MR6_NR3", 6, 3, gemm_blis_generic},
    {"MR6_NR2", 6, 2, gemm_blis_generic},
    {"MR5_NR16", 5, 16, gemm_blis_generic},
    {"MR5_NR15", 5, 15, gemm_blis_generic},
    {"MR5_NR14", 5, 14, gemm_blis_generic},
    {"MR5_NR13", 5, 13, gemm_blis_generic},
    {"MR5_NR12", 5, 12, gemm_blis_generic},
    {"MR5_NR11", 5, 11, gemm_blis_generic},
    {"MR5_NR10", 5, 10, gemm_blis_generic},
    {"MR5_NR9", 5, 9, gemm_blis_generic},
    {"MR5_NR8", 5, 8, gemm_blis_generic},
    {"MR5_NR7", 5, 7, gemm_blis_generic},
    {"MR5_NR6", 5, 6, gemm_blis_generic},
    {"MR5_NR5", 5, 5, gemm_blis_generic},
    {"MR5_NR4", 5, 4, gemm_blis_generic},
    {"MR5_NR3", 5, 3, gemm_blis_generic},
    {"MR5_NR2", 5, 2, gemm_blis_generic},
    {"MR4_NR16", 4, 16, gemm_blis_generic},
    {"MR4_NR15", 4, 15, gemm_blis_generic},
    {"MR4_NR14", 4, 14, gemm_blis_generic},
    {"MR4_NR13", 4, 13, gemm_blis_generic},
    {"MR4_NR12", 4, 12, gemm_blis_generic},
    {"MR4_NR11", 4, 11, gemm_blis_generic},
    {"MR4_NR10", 4, 10, gemm_blis_generic},
    {"MR4_NR9", 4, 9, gemm_blis_generic},
    {"MR4_NR8", 4, 8, gemm_blis_generic},
    {"MR4_NR7", 4, 7, gemm_blis_generic},
    {"MR4_NR6", 4, 6, gemm_blis_generic},
    {"MR4_NR5", 4, 5, gemm_blis_generic},
    {"MR4_NR4", 4, 4, gemm_blis_generic},
    {"MR4_NR3", 4, 3, gemm_blis_generic},
    {"MR4_NR2", 4, 2, gemm_blis_generic},
    {"MR3_NR16", 3, 16, gemm_blis_generic},
    {"MR3_NR15", 3, 15, gemm_blis_generic},
    {"MR3_NR14", 3, 14, gemm_blis_generic},
    {"MR3_NR13", 3, 13, gemm_blis_generic},
    {"MR3_NR12", 3, 12, gemm_blis_generic},
    {"MR3_NR11", 3, 11, gemm_blis_generic},
    {"MR3_NR10", 3, 10, gemm_blis_generic},
    {"MR3_NR9", 3, 9, gemm_blis_generic},
    {"MR3_NR8", 3, 8, gemm_blis_generic},
    {"MR3_NR7", 3, 7, gemm_blis_generic},
    {"MR3_NR6", 3, 6, gemm_blis_generic},
    {"MR3_NR5", 3, 5, gemm_blis_generic},
    {"MR3_NR4", 3, 4, gemm_blis_generic},
    {"MR3_NR3", 3, 3, gemm_blis_generic},
    {"MR3_NR2", 3, 2, gemm_blis_generic},
    {"MR2_NR16", 2, 16, gemm_blis_generic},
    {"MR2_NR15", 2, 15, gemm_blis_generic},
    {"MR2_NR14", 2, 14, gemm_blis_generic},
    {"MR2_NR13", 2, 13, gemm_blis_generic},
    {"MR2_NR12", 2, 12, gemm_blis_generic},
    {"MR2_NR11", 2, 11, gemm_blis_generic},
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

static const int num_variants = 465;

#ifdef __cplusplus
}
#endif
