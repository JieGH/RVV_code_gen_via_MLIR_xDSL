#ifndef GENERATED_KERNELS_H
#define GENERATED_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Cache config: MC=192, NC=1365, KC=512
void gemm_macro_v_fp32_MR4_NR4(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

// Cache config: MC=192, NC=1365, KC=512
void gemm_macro_v_fp32_MR4_NR8(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

// Cache config: MC=384, NC=1536, KC=256
void gemm_macro_v_fp32_MR8_NR8(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

typedef void (*gemm_macro_fn)(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

struct KernelVariant {
    const char* name;
    int mr;
    int nr;
    int mc;
    int nc;
    int kc;
    gemm_macro_fn func;
};

static const KernelVariant generated_variants[] = {
    {"MR4_NR4", 4, 4, 192, 1365, 512, gemm_macro_v_fp32_MR4_NR4},
    {"MR4_NR8", 4, 8, 192, 1365, 512, gemm_macro_v_fp32_MR4_NR8},
    {"MR8_NR8", 8, 8, 384, 1536, 256, gemm_macro_v_fp32_MR8_NR8},
};

static const int num_variants = 3;

#ifdef __cplusplus
}
#endif

#endif // GENERATED_KERNELS_H
