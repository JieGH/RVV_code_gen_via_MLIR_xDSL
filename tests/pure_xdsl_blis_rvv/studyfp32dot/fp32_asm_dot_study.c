#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MR 4
#define NR 16
#define KR 4

#define Mcol(r,c)  M[(c)*(ldM) + (r)]
#define Ccol(r,c)  Cr[(c)*(ldC) + (r)]

void pack_A_fp32_asm(int mc, int kc, const float *M, int ldM, float *Mc, int MR_) {
    int p = 0;
    for (int i = 0; i < mc; i += MR_) {
        int mr = (mc - i < MR_) ? (mc - i) : MR_;
        for (int j = 0; j < kc; j += KR) {
            int kr = (kc - j < KR) ? (kc - j) : KR;
            for (int ii = 0; ii < mr; ii++) {
                for (int jj = 0; jj < kr; jj++) Mc[p++] = Mcol(i + ii, j + jj);
                for (int jj = kr; jj < KR; jj++) Mc[p++] = 0.0f;
            }
            for (int ii = mr; ii < MR_; ii++)
                for (int jj = 0; jj < KR; jj++) Mc[p++] = 0.0f;
        }
    }
}

void pack_B_fp32_asm(int kc, int nc, const float *M, int ldM, float *Mc, int NR_) {
    int p = 0;
    for (int j = 0; j < nc; j += NR_) {
        int nr = (nc - j < NR_) ? (nc - j) : NR_;
        for (int i = 0; i < kc; i += KR) {
            int kr = (kc - i < KR) ? (kc - i) : KR;
            for (int ii = 0; ii < nr; ii++) {
                for (int jj = 0; jj < kr; jj++) Mc[p++] = Mcol(i + jj, j + ii);
                for (int jj = kr; jj < KR; jj++) Mc[p++] = 0.0f;
            }
            for (int ii = nr; ii < NR_; ii++)
                for (int jj = 0; jj < KR; jj++) Mc[p++] = 0.0f;
        }
    }
}

#if defined(__riscv_v) || defined(__riscv_vector) || defined(__riscv)
__attribute__((noinline))
void microkernel_asm_dot_fp32(int mr, int nr, int kc,
                              const float *Ar, const float *Br,
                              float *Cr, float beta, int ldC)
{
    int kc_it = kc / KR;
    int ldC_bytes = ldC * sizeof(float);
    
    // Convert beta to an integer representation to check for 0.0f easily
    // Or we can just pass it in fa0 and use fclass.
    
    asm volatile (
        "vsetvli t0, %[nr], e32, m2, ta, ma \n"

        // Initialize vC0, vC1, vC2, vC3 (v0, v2, v4, v6)
        "fmv.x.w t1, %[beta] \n"
        "beqz t1, 1f \n"
        
        // Beta != 0.0
        "mv t2, %[Cr] \n"
        "vlse32.v v0, (t2), %[ldcb] \n"
        "addi t2, t2, 4 \n" // Cr + 1 float = +4 bytes
        "vlse32.v v2, (t2), %[ldcb] \n"
        "addi t2, t2, 4 \n"
        "vlse32.v v4, (t2), %[ldcb] \n"
        "addi t2, t2, 4 \n"
        "vlse32.v v6, (t2), %[ldcb] \n"
        
        "lui t3, 0x3f800 \n" // 1.0f in hex is 0x3f800000
        "beq t1, t3, 2f \n"  // if beta == 1.0, skip multiply
        
        "vfmul.vf v0, v0, %[beta] \n"
        "vfmul.vf v2, v2, %[beta] \n"
        "vfmul.vf v4, v4, %[beta] \n"
        "vfmul.vf v6, v6, %[beta] \n"
        "j 2f \n"

        "1: \n"
        // Beta == 0.0
        "vmv.v.i v0, 0 \n"
        "vmv.v.i v2, 0 \n"
        "vmv.v.i v4, 0 \n"
        "vmv.v.i v6, 0 \n"

        "2: \n"
        // K-loop
        "mv t1, %[kc_it] \n"
        "mv t2, %[Ar] \n"
        "mv t3, %[Br] \n"
        "li t4, 16 \n" // KR * 4 = 16 (stride)

        "3: \n" // Loop start
        
        // l = 0
        "vlse32.v v8, (t3), t4 \n"
        "flw ft0, 0(t2) \n"
        "flw ft1, 16(t2) \n"
        "flw ft2, 32(t2) \n"
        "flw ft3, 48(t2) \n"
        "vfmacc.vf v0, ft0, v8 \n"
        "vfmacc.vf v2, ft1, v8 \n"
        "vfmacc.vf v4, ft2, v8 \n"
        "vfmacc.vf v6, ft3, v8 \n"
        
        // l = 1
        "addi t5, t3, 4 \n"
        "vlse32.v v8, (t5), t4 \n"
        "flw ft0, 4(t2) \n"
        "flw ft1, 20(t2) \n"
        "flw ft2, 36(t2) \n"
        "flw ft3, 52(t2) \n"
        "vfmacc.vf v0, ft0, v8 \n"
        "vfmacc.vf v2, ft1, v8 \n"
        "vfmacc.vf v4, ft2, v8 \n"
        "vfmacc.vf v6, ft3, v8 \n"

        // l = 2
        "addi t5, t3, 8 \n"
        "vlse32.v v8, (t5), t4 \n"
        "flw ft0, 8(t2) \n"
        "flw ft1, 24(t2) \n"
        "flw ft2, 40(t2) \n"
        "flw ft3, 56(t2) \n"
        "vfmacc.vf v0, ft0, v8 \n"
        "vfmacc.vf v2, ft1, v8 \n"
        "vfmacc.vf v4, ft2, v8 \n"
        "vfmacc.vf v6, ft3, v8 \n"

        // l = 3
        "addi t5, t3, 12 \n"
        "vlse32.v v8, (t5), t4 \n"
        "flw ft0, 12(t2) \n"
        "flw ft1, 28(t2) \n"
        "flw ft2, 44(t2) \n"
        "flw ft3, 60(t2) \n"
        "vfmacc.vf v0, ft0, v8 \n"
        "vfmacc.vf v2, ft1, v8 \n"
        "vfmacc.vf v4, ft2, v8 \n"
        "vfmacc.vf v6, ft3, v8 \n"

        "addi t2, t2, 64 \n"  // Ar += MR*KR*4
        "addi t3, t3, 256 \n" // Br += NR*KR*4

        "addi t1, t1, -1 \n"
        "bnez t1, 3b \n"

        // Write back to C (beta == 0 is assumed for now in this study, let's just save)
        "mv t2, %[Cr] \n"
        "vsse32.v v0, (t2), %[ldcb] \n"
        "addi t2, t2, 4 \n"
        "vsse32.v v2, (t2), %[ldcb] \n"
        "addi t2, t2, 4 \n"
        "vsse32.v v4, (t2), %[ldcb] \n"
        "addi t2, t2, 4 \n"
        "vsse32.v v6, (t2), %[ldcb] \n"

        : 
        : [nr] "r" (nr),
          [beta] "f" (beta),
          [Cr] "r" (Cr),
          [ldcb] "r" (ldC_bytes),
          [kc_it] "r" (kc_it),
          [Ar] "r" (Ar),
          [Br] "r" (Br)
        : "t0", "t1", "t2", "t3", "t4", "t5",
          "ft0", "ft1", "ft2", "ft3",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"
    );
}
#endif

void gemm_asm_fp32_dot(int m, int n, int k,
                       const float *A, int ldA,
                       const float *B, int ldB,
                       float beta,
                       float *C, int ldC,
                       int MC, int NC, int KC)
{
    int MC_pad = ((MC + MR - 1) / MR) * MR;
    int KC_pad = ((KC + KR - 1) / KR) * KR;
    int NC_pad = ((NC + NR - 1) / NR) * NR;

    float *Ac = (float *)malloc((size_t)MC_pad * KC_pad * sizeof(float));
    float *Bc = (float *)malloc((size_t)KC_pad * NC_pad * sizeof(float));

    float one = 1.0f, betaI;

    for (int jc = 0; jc < n; jc += NC) {
        int nc = (n - jc < NC) ? (n - jc) : NC;
        for (int pc = 0; pc < k; pc += KC) {
            int kc = (k - pc < KC) ? (k - pc) : KC;
            int kc_pad = ((kc + KR - 1) / KR) * KR;

            pack_B_fp32_asm(kc, nc, &B[pc + jc * ldB], ldB, Bc, NR);

            betaI = (pc == 0) ? beta : one;

            for (int ic = 0; ic < m; ic += MC) {
                int mc = (m - ic < MC) ? (m - ic) : MC;

                pack_A_fp32_asm(mc, kc, &A[ic + pc * ldA], ldA, Ac, MR);

                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = (nc - jr < NR) ? (nc - jr) : NR;
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = (mc - ir < MR) ? (mc - ir) : MR;

                        const float *Ar = &Ac[ir * kc_pad];
                        const float *Br = &Bc[jr * kc_pad];
                        float *Cr = &C[(ic + ir) + (jc + jr) * ldC];

                        microkernel_asm_dot_fp32(mr, nr, kc_pad, Ar, Br, Cr, betaI, ldC);
                    }
                }
            }
        }
    }

    free(Ac);
    free(Bc);
}

void gemm_fp32_naive(int m, int n, int k,
                     const float *A, int ldA,
                     const float *B, int ldB,
                     float beta,
                     float *C, int ldC)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++)
                sum += A[i + p * ldA] * B[p + j * ldB];
            C[i + j * ldC] = (beta * C[i + j * ldC]) + sum;
        }
}

int main() {
    int m = 256, n = 256, k = 256;
    int lda = m, ldb = k, ldc = m;
    float *A = (float *)malloc(m * k * sizeof(float));
    float *B = (float *)malloc(k * n * sizeof(float));
    float *C_naive = (float *)malloc(m * n * sizeof(float));
    float *C_asm = (float *)malloc(m * n * sizeof(float));

    for (int i = 0; i < m * k; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < k * n; i++) B[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < m * n; i++) {
        C_naive[i] = 0.0f;
        C_asm[i] = 0.0f;
    }

    gemm_fp32_naive(m, n, k, A, lda, B, ldb, 0.0f, C_naive, ldc);
    gemm_asm_fp32_dot(m, n, k, A, lda, B, ldb, 0.0f, C_asm, ldc, 128, 128, 128);

    float max_err = 0.0f;
    for (int i = 0; i < m * n; i++) {
        float err = fabsf(C_naive[i] - C_asm[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max absolute error (ASM vs Naive): %e  ->  %s\n", max_err, max_err < 1e-3 ? "PASS" : "FAIL");

    if (max_err >= 1e-3) {
        free(A); free(B); free(C_naive); free(C_asm);
        return 1;
    }

    printf("\n--- Performance Benchmarks (>= 3.0 seconds each) ---\n");
    
    struct timespec start, end;
    double elapsed = 0.0;
    int iters = 1;
    while (elapsed < 3.0) {
        iters *= 2;
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int i=0; i<iters; i++) {
            gemm_asm_fp32_dot(m, n, k, A, lda, B, ldb, 0.0f, C_asm, ldc, 128, 128, 128);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    }
    double gflops_asm = (2.0 * m * n * k * iters) / (elapsed * 1e9);
    printf("ASM FP32 DOT:    %d iters in %.2f s => %.2f GFLOPS\n", iters, elapsed, gflops_asm);

    free(A); free(B); free(C_naive); free(C_asm);
    return 0;
}
