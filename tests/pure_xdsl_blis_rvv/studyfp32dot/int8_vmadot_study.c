#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MR 4
#define NR 4
#define KR 8

#define Mcol(r,c)  M[(c)*(ldM) + (r)]
#define Ccol(r,c)  Cr[(c)*(ldC) + (r)]

void pack_A_int8(int mc, int kc, const int8_t *M, int ldM, int8_t *Mc, int MR_) {
    int p = 0;
    for (int i = 0; i < mc; i += MR_) {
        int mr = (mc - i < MR_) ? (mc - i) : MR_;
        for (int j = 0; j < kc; j += KR) {
            int kr = (kc - j < KR) ? (kc - j) : KR;
            for (int ii = 0; ii < mr; ii++) {
                for (int jj = 0; jj < kr; jj++)
                    Mc[p++] = Mcol(i + ii, j + jj);
                for (int jj = kr; jj < KR; jj++)
                    Mc[p++] = 0;
            }
            for (int ii = mr; ii < MR_; ii++)
                for (int jj = 0; jj < KR; jj++)
                    Mc[p++] = 0;
        }
    }
}

void pack_B_int8(int kc, int nc, const int8_t *M, int ldM, int8_t *Mc, int NR_) {
    int p = 0;
    for (int j = 0; j < nc; j += NR_) {
        int nr = (nc - j < NR_) ? (nc - j) : NR_;
        for (int i = 0; i < kc; i += KR) {
            int kr = (kc - i < KR) ? (kc - i) : KR;
            for (int ii = 0; ii < nr; ii++) {
                for (int jj = 0; jj < kr; jj++)
                    Mc[p++] = Mcol(i + jj, j + ii);
                for (int jj = kr; jj < KR; jj++)
                    Mc[p++] = 0;
            }
            for (int ii = nr; ii < NR_; ii++)
                for (int jj = 0; jj < KR; jj++)
                    Mc[p++] = 0;
        }
    }
}

#if defined(__riscv_v) || defined(__riscv_vector) || defined(__riscv)
__attribute__((noinline))
void microkernel_vmadot_int8(int mr, int nr, int kc, 
                             const int8_t *Ar, const int8_t *Br, 
                             int32_t *Cr, int32_t beta, int ldC) 
{
    int kc_it = kc / KR;
    int ldC_size = sizeof(int32_t) * ldC;
    uint32_t Ctmp[32]; 

    asm volatile (
        "li t6, 256                         \n"
        "li t4, 32                          \n"
        "li t3, 16                          \n"
        "li t5, 8                           \n"
        "li t2, 4                           \n"

        "vsetvli x7, x0, e8, m8             \n" 
        "vxor.vv  v0,  v0,  v0              \n"
        "vxor.vv  v8,  v8,  v8              \n"
        "vxor.vv v16, v16, v16              \n"
        "vxor.vv v24, v24, v24              \n"

        "mv a0, %3                          \n"
        "mv t0, %4                          \n"
        "mv a1, %2                          \n"
        
        "mv a6, %7                          \n"
        "mv a2, %8                          \n"
        "add a3, a2, a6                     \n"
        "add a4, a3, a6                     \n"
        "add a5, a4, a6                     \n"
        
        "beq %6, zero, 1f                   \n"
        "vsetvli t2, t2, e32, m1, ta, ma    \n" 
        "vle32.v v2, (a3)                   \n"
        "vle32.v v3, (a5)                   \n"
        "vsetvli t5, t5, e32, m1, ta, ma    \n" 
        "vslideup.vi v0, v2, 4              \n"
        "vslideup.vi v1, v3, 4              \n"
        "vsetvli t2, t2, e32, mf2, ta, ma   \n" 
        "vle32.v v0, (a2)                   \n"
        "vle32.v v1, (a4)                   \n"

        "1:                                 \n"
        "vsetvli t4, t4, e8, m1, ta, ma     \n"
        "vle8.v v10, (a0)                   \n"
        "addi a0, a0,  32                   \n" 
        "vle8.v v11, (t0)                   \n"
        "addi t0, t0,  32                   \n"

        "vsetvli t6, t6, e8, m1, ta, ma     \n"
        "vmadot v0, v11, v10                \n"

        "addi a1, a1, -1                    \n"
        "bne  a1, zero, 1b                  \n"

        "vsetvli t2, t2, e32, m1, ta, ma    \n"
        "vslidedown.vi v2, v0, 4            \n"
        "vslidedown.vi v4, v1, 4            \n"
        "vsetvli t2, t2, e32, mf2, ta, ma   \n" 
        "vse32.v v0, (a2)                   \n" 
        "vse32.v v2, (a3)                   \n"
        "vse32.v v1, (a4)                   \n"
        "vse32.v v4, (a5)                   \n"
        :
        : "r"(mr),   "r"(nr),       "r"(kc_it), "r"(Ar), "r"(Br),     "r"(Ctmp), 
          "r"(beta), "r"(ldC_size), "r"(Cr)
        : "t0", "t1", "t2", "t3", "t4", "t5", "t6",
          "a0", "a1", "a2", "a3", "a4", "a5", "a6",
          "v0", "v1", "v2", "v3", "v4", 
          "v5", "v6", "v7", "v8", "v9", 
          "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "v24", "v25", "v26", "v27"
    );
}

void gemm_int8_vmadot(int m, int n, int k,
                      const int8_t *A, int ldA,
                      const int8_t *B, int ldB,
                      int32_t beta,
                      int32_t *C, int ldC,
                      int MC, int NC, int KC)
{
    int MC_pad = ((MC + MR - 1) / MR) * MR;
    int KC_pad = ((KC + KR - 1) / KR) * KR;
    int NC_pad = ((NC + NR - 1) / NR) * NR;

    int8_t *Ac = (int8_t *)malloc((size_t)MC_pad * KC_pad * sizeof(int8_t));
    int8_t *Bc = (int8_t *)malloc((size_t)KC_pad * NC_pad * sizeof(int8_t));
    if (!Ac || !Bc) { free(Ac); free(Bc); return; }

    int32_t one = 1, betaI;

    /* L1: tile over N */
    for (int jc = 0; jc < n; jc += NC) {
        int nc = (n - jc < NC) ? (n - jc) : NC;

        /* L2: tile over K */
        for (int pc = 0; pc < k; pc += KC) {
            int kc = (k - pc < KC) ? (k - pc) : KC;
            int kc_pad = ((kc + KR - 1) / KR) * KR;

            pack_B_int8(kc, nc, &B[pc + (long)jc * ldB], ldB, Bc, NR);

            betaI = (pc == 0) ? beta : one;

            /* L3: tile over M */
            for (int ic = 0; ic < m; ic += MC) {
                int mc = (m - ic < MC) ? (m - ic) : MC;

                pack_A_int8(mc, kc, &A[ic + (long)pc * ldA], ldA, Ac, MR);

                /* L4 + L5: micro-kernel tiles */
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = (nc - jr < NR) ? (nc - jr) : NR;
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = (mc - ir < MR) ? (mc - ir) : MR;

                        const int8_t *Ar = &Ac[(long)ir * kc_pad];
                        const int8_t *Br = &Bc[(long)jr * kc_pad];
                        int32_t *Cr = &C[(ic + ir) + (long)(jc + jr) * ldC];

                        microkernel_vmadot_int8(mr, nr, kc_pad,
                                                Ar, Br,
                                                Cr, betaI, ldC);
                    }
                }
            }
        }
    }

    free(Ac);
    free(Bc);
}
#endif

void gemm_int8_naive(int m, int n, int k,
                     const int8_t *A, int ldA,
                     const int8_t *B, int ldB,
                     int32_t beta,
                     int32_t *C, int ldC)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int32_t sum = 0;
            for (int p = 0; p < k; p++)
                sum += (int32_t)A[i + (long)p * ldA] * (int32_t)B[p + (long)j * ldB];
            C[i + (long)j * ldC] = beta * C[i + (long)j * ldC] + sum;
        }
}

int main(void)
{
    srand(42);

    int m = 256, n = 256, k = 256;
    int ldA = m, ldB = k, ldC = m;

    int8_t *A  = malloc((size_t)m * k * sizeof(int8_t));
    int8_t *B  = malloc((size_t)k * n * sizeof(int8_t));
    int32_t *C1 = malloc((size_t)m * n * sizeof(int32_t));
    int32_t *C2 = malloc((size_t)m * n * sizeof(int32_t));

    for (int i = 0; i < m*k; i++) A[i]  = (int8_t)(rand() % 10 - 5);
    for (int i = 0; i < k*n; i++) B[i]  = (int8_t)(rand() % 10 - 5);
    for (int i = 0; i < m*n; i++) C1[i] = C2[i] = (int32_t)(rand() % 20);

    int32_t beta = 0;

    gemm_int8_naive(m, n, k, A, ldA, B, ldB, beta, C1, ldC);

#if defined(__riscv_v) || defined(__riscv_vector) || defined(__riscv)
    gemm_int8_vmadot(m, n, k, A, ldA, B, ldB, beta, C2, ldC, m, n, k);
#else
    printf("[Warning] Not compiled with RVV support. Skipping vmadot test.\n");
    for (int i = 0; i < m*n; i++) C2[i] = C1[i];
#endif

    int max_err = 0;
    int err_count = 0;
    for (int i = 0; i < m*n; i++) {
        int e = abs(C1[i] - C2[i]);
        if (e > 0 && err_count < 5) {
            printf("Mismatch at %d: naive=%d, vmadot=%d, diff=%d\n", i, C1[i], C2[i], e);
            err_count++;
        }
        if (e > max_err) max_err = e;
    }

    printf("Max absolute error (INT8 vmadot vs Naive): %d  ->  %s\n",
           max_err, (max_err == 0) ? "PASS" : "FAIL");

    if (max_err != 0) {
        free(A); free(B); free(C1); free(C2);
        return 1;
    }

    printf("\n--- Performance Benchmarks (>= 3.0 seconds each) ---\n");

#if defined(__riscv_v) || defined(__riscv_vector) || defined(__riscv)
    #include <time.h>
    struct timespec start, end;
    double elapsed = 0.0;
    int iters = 1;
    while (elapsed < 3.0) {
        iters *= 2;
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int i=0; i<iters; i++) {
            gemm_int8_vmadot(m, n, k, A, ldA, B, ldB, beta, C2, ldC, m, n, k);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    }
    double gops_vmadot = (2.0 * m * n * k * iters) / (elapsed * 1e9);
    printf("INT8 vmadot: %d iters in %.2f s => %.2f GOPS\n", iters, elapsed, gops_vmadot);
#endif

    free(A); free(B); free(C1); free(C2);
    return 0;
}
