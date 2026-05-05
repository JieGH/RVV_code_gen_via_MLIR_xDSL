#ifndef GEMM_MACRO_DISPATCH_H
#define GEMM_MACRO_DISPATCH_H

#include <cstddef>
#include <cstdio>
#include <cstdlib>

// Forward declarations for types used in gemm_blis.h
typedef struct cntx_s cntx_t;
typedef struct auxinfo_s auxinfo_t;
typedef void (*gemm_ukr_ft)(int mr, int nr, int kc, const void *alpha,
                            const void *a, const void *b, const void *beta,
                            void *c, int inc_c, int ld_c, const void *aux,
                            const void *cntx);

extern "C" {
/**
 * Specialized GEMM Macro-Kernel generated via xDSL.
 * Layout: A=ColumnMajor/NoTrans, B=ColumnMajor/NoTrans, C=ColumnMajor
 */
void gemm_blis_v_CN_CN_C_fp32(int m, int n, int k, float alpha, float *A,
                              int ldA, float *B, int ldB, float beta, float *C,
                              int ldC, float *Ac, float *Bc);
}

/**
 * Universal Dispatcher matching the UPV Benchmark signature.
 * Intercepts the call to gemm_blis_B3A2C0 and routes it to the specialized xDSL
 * implementation.
 */
inline void gemm_blis_B3A2C0(char orderA, char orderB, char orderC, char transA,
                             char transB, size_t m, size_t n, size_t k,
                             float alpha, float *A, size_t ldA, float *B,
                             size_t ldB, float beta, float *C, size_t ldC,
                             float *Ac, float *Bc, size_t MC, size_t NC,
                             size_t KC, const cntx_t *cntx, auxinfo_t *aux,
                             gemm_ukr_ft gemm_kernel) {
  // Dispatch to vectorized macro kernel if layout matches compile-time
  // specialization
  if (orderA == 'C' && transA == 'N' && orderB == 'C' && transB == 'N' &&
      orderC == 'C') {
    gemm_blis_v_CN_CN_C_fp32((int)m, (int)n, (int)k, alpha, A, (int)ldA, B,
                             (int)ldB, beta, C, (int)ldC, Ac, Bc);
  } else {
    // Layout mismatch fallback
    fprintf(stderr,
            "xDSL Macro-Kernel Error: Layout Combination (A:%c%c, B:%c%c, "
            "C:%c) is not yet specialized.\n",
            orderA, transA, orderB, transB, orderC);
    fprintf(stderr, "Please generate the specialized variant using "
                    "macro_builder.py first.\n");
    exit(-1);
  }
}

#endif // GEMM_MACRO_DISPATCH_H
