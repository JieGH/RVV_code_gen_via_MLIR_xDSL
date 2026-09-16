#pragma once

#include <stddef.h>
#include <stdint.h>
#include "perf_counters.h"

#define CONFIG_DATATYPE "fp32"
#define CONFIG_METHOD "AXPY_FP32_Pack"
#define CONFIG_VLEN 256
#define CONFIG_ARCH "jbpi2"
#define CONFIG_ITERATIONS 1000

#define SKIP_REF_RUN 1

struct NcOverrideEntry { const char* prefix; int nc; };
static const NcOverrideEntry CONFIG_NC_OVERRIDES[] = { {"B3", 128} };
static const size_t CONFIG_NC_OVERRIDES_COUNT = 1;

#ifdef __cplusplus
extern "C" {
#endif

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR15(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR15(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR15(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR15(float*, float*, size_t, size_t, size_t);
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR16(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR16(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR16(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR16(float*, float*, size_t, size_t, size_t);

void gemm_blis_generic(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc, int mr, int nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack, perf_gemm_counters* perf);

typedef void (*gemm_blis_fn)(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc, int mr, int nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack, perf_gemm_counters* perf);

struct KernelVariant {
    const char* name;
    int mr;
    int nr;
    gemm_blis_fn func;
};

static const KernelVariant generated_variants[] = {
    {"MR16_NR15", 16, 15, gemm_blis_generic},
};

static const int num_variants = 1;

#ifdef __cplusplus
}
#endif
