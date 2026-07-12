#ifndef GENERATED_KERNELS_H
#define GENERATED_KERNELS_H

#define CONFIG_DATATYPE "fp32"
#define CONFIG_METHOD "AXPY_FP32_Pack"
#define CONFIG_VLEN 256
#define CONFIG_ARCH "jbpi2"

#ifdef __cplusplus
extern "C" {
#endif

void gemm_macro_v_fp32_MR4_NR4(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc);

void gemm_macro_v_fp32_MR2_NR2(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc);

typedef void (*gemm_macro_fn)(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc);

struct KernelVariant {
    const char* name;
    int mr;
    int nr;
    gemm_macro_fn func;
};

static const KernelVariant generated_variants[] = {
    {"MR4_NR4", 4, 4, gemm_macro_v_fp32_MR4_NR4},
    {"MR2_NR2", 2, 2, gemm_macro_v_fp32_MR2_NR2},
};

static const int num_variants = 2;

#ifdef __cplusplus
}
#endif

#endif // GENERATED_KERNELS_H
