#ifndef GENERATED_KERNELS_H
#define GENERATED_KERNELS_H

#define CONFIG_DATATYPE "fp32"
#define CONFIG_METHOD "AXPY_FP32_Pack"
#define CONFIG_VLEN 256
#define CONFIG_ARCH "jbpi2"

#ifdef __cplusplus
extern "C" {
#endif

// Cache config: MC=192, NC=1365, KC=512
void gemm_macro_v_fp32_MR4_NR4(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

// Cache config: MC=192, NC=1365, KC=512
void gemm_macro_v_fp32_MR4_NR6(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

// Cache config: MC=288, NC=1592, KC=341
void gemm_macro_v_fp32_MR6_NR6(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

// Cache config: MC=192, NC=1365, KC=512
void gemm_macro_v_fp32_MR4_NR8(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

// Cache config: MC=288, NC=1592, KC=341
void gemm_macro_v_fp32_MR6_NR8(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

// Cache config: MC=384, NC=1536, KC=256
void gemm_macro_v_fp32_MR8_NR8(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

// Cache config: MC=768, NC=1024, KC=128
void gemm_macro_v_fp32_MR16_NR16(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc);

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
    {"MR4_NR6", 4, 6, 192, 1365, 512, gemm_macro_v_fp32_MR4_NR6},
    {"MR6_NR6", 6, 6, 288, 1592, 341, gemm_macro_v_fp32_MR6_NR6},
    {"MR4_NR8", 4, 8, 192, 1365, 512, gemm_macro_v_fp32_MR4_NR8},
    {"MR6_NR8", 6, 8, 288, 1592, 341, gemm_macro_v_fp32_MR6_NR8},
    {"MR8_NR8", 8, 8, 384, 1536, 256, gemm_macro_v_fp32_MR8_NR8},
    {"MR16_NR16", 16, 16, 768, 1024, 128, gemm_macro_v_fp32_MR16_NR16},
};

static const int num_variants = 7;

#ifdef __cplusplus
}
#endif

#endif // GENERATED_KERNELS_H
