/*
   LDX Reference Kernel Benchmark — MACROKERNEL
   ----------------------------------------------
   Benchmarks the FULL problem size (m x n x k) by calling the
   reference macrokernel `gemm_blis_B3A2C0`.
   This includes all packing and cache-blocking overhead, matching
   exactly how the pure xDSL driver_benchmark measures performance.

   Run from the test_ext_ldx_macc directory; reads datasets/testCase_repeat.dat.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "dtypes.h"
#include "exo_matrix_RVV_fp32.h"
#include "model_level.h"
#include "gemm_blis.h"

#undef min
#undef max
#define mymin(a,b)    ( (a) < (b) ? (a) : (b) )
#define dabs(a)       ( (a) > 0.0f ? (a) : -(a) )
#define Ccol(i,j)     C[(j)*(ldC)+(i)]
#define Cgcol(i,j)    Cg[(j)*(ldC)+(i)]

/* ------------------------------------------------------------------ */
/*  Timer (ns resolution)                                             */
/* ------------------------------------------------------------------ */
static double dclock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* ------------------------------------------------------------------ */
/*  Aligned alloc helpers                                              */
/* ------------------------------------------------------------------ */
static void *my_aligned_alloc(size_t alignment, size_t size) {
    size_t total = size + alignment;
    char *buf = (char *)calloc(1, total);
    if (!buf) { perror("calloc"); exit(1); }
    size_t off = alignment - ((size_t)buf % alignment);
    char *ret = buf + off;
    *(unsigned char *)(ret - 1) = (unsigned char)off;
    return ret;
}
static void my_aligned_free(void *ptr) {
    unsigned char off = *(((unsigned char *)ptr) - 1);
    free(((char *)ptr) - off);
}

/* ------------------------------------------------------------------ */
/*  Naive reference GEMM (col-major, NN)                              */
/* ------------------------------------------------------------------ */
static void gemm_ref(int m, int n, int k,
                     DTYPE alpha, const DTYPE *A, int ldA,
                     const DTYPE *B, int ldB,
                     DTYPE beta, DTYPE *C, int ldC) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            DTYPE acc = 0.0f;
            for (int p = 0; p < k; p++)
                acc += A[p * ldA + i] * B[j * ldB + p];
            C[j * ldC + i] = alpha * acc + beta * C[j * ldC + i];
        }
}

/* ================================================================== */
int main(void) {
    const int UKR_M  = 16;
    const int UKR_N  = 15;
    const int N_REPS = 1000;
    const float ERRORTHD = 1.0e-3f;

    /* Build the dispatch table */
    ukrFunction ****ukrmatrix = allocateMatrix();
    fillMatrix(ukrmatrix);

    FILE *fp = fopen("datasets/testCase_repeat.dat", "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot open datasets/testCase_repeat.dat\n");
        return 1;
    }

    printf("\n");
    printf("  %-24s | %6s GOPs  "
           "[min, p5, med, p95, max | iqr, std, CV%%, mx/md]  Status\n",
           "Layer (M x N x K)", "Median");
    printf("  %-24s-+-%s\n",
           "------------------------", 
           "-----------------------------------------------------------------------");

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        char label[128];
        int m, n, k;
        if (sscanf(line, "%127s %d %d %d", label, &m, &n, &k) != 4) continue;

        int ldA = m, ldB = k, ldC = m;

        /* Allocate raw matrices */
        DTYPE *A  = (DTYPE *)my_aligned_alloc(64, (size_t)m * k * sizeof(DTYPE));
        DTYPE *B  = (DTYPE *)my_aligned_alloc(64, (size_t)k * n * sizeof(DTYPE));
        DTYPE *C  = (DTYPE *)my_aligned_alloc(64, (size_t)m * n * sizeof(DTYPE));
        DTYPE *Cg = (DTYPE *)my_aligned_alloc(64, (size_t)m * n * sizeof(DTYPE));

        /* Macrokernel requires its own packed buffers */
        int mc, nc, kc;
        get_optim_mc_nc_kc(sizeof(DTYPE), m, n, k, UKR_M, UKR_N, &mc, &nc, &kc);
        DTYPE *Ac = (DTYPE *)my_aligned_alloc(64, (size_t)(UKR_M + mc) * (UKR_M + kc) * sizeof(DTYPE));
        DTYPE *Bc = (DTYPE *)my_aligned_alloc(64, (size_t)(UKR_N + kc) * (UKR_N + nc) * sizeof(DTYPE));

        /* Fill with random data */
        srand(42);
        for (int i = 0; i < m * k; i++) A[i] = (DTYPE)rand() / RAND_MAX + 1.0f;
        for (int i = 0; i < k * n; i++) B[i] = (DTYPE)rand() / RAND_MAX + 1.0f;
        for (int i = 0; i < m * n; i++) C[i] = 0.0f;
        memcpy(Cg, C, (size_t)m * n * sizeof(DTYPE));

        float alpha = 1.0f, beta = 0.0f;

        /* ---- Correctness Check ---- */
        gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N',
                         m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC,
                         Ac, Bc, mc, nc, kc,
                         NULL, NULL, NULL, ukrmatrix);

        gemm_ref(m, n, k, alpha, A, ldA, B, ldB, beta, Cg, ldC);

        int correct = 1;
        for (int j = 0; j < n && correct; j++)
            for (int i = 0; i < m && correct; i++)
                if (dabs(Ccol(i, j) - Cgcol(i, j)) > ERRORTHD)
                    correct = 0;

        /* ---- Warmup (Crucial for cache) ---- */
        for (int w = 0; w < 5; w++) {
            gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N',
                             m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC,
                             Ac, Bc, mc, nc, kc,
                             NULL, NULL, NULL, ukrmatrix);
        }

        /* ---- Performance Benchmark ---- */
        double rep_gops[1000];
        double sum_gops = 0.0;

        for (int r = 0; r < N_REPS; r++) {
            /* Reset C for safety */
            memset(C, 0, (size_t)m * n * sizeof(DTYPE));

            double t1 = dclock_ns();
            gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N',
                             m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC,
                             Ac, Bc, mc, nc, kc,
                             NULL, NULL, NULL, ukrmatrix);
            double t2 = dclock_ns();

            double t = t2 - t1;
            /* Full M x N x K mathematics */
            double gops = (t > 0.0) ? (2.0 * m * n * k) / (t * 1e9) : 0.0;
            rep_gops[r] = gops;
            sum_gops += gops;
        }

        /* Sort percentiles */
        for (int i = 0; i < N_REPS - 1; i++)
            for (int j = 0; j < N_REPS - i - 1; j++)
                if (rep_gops[j] > rep_gops[j+1]) {
                    double tmp = rep_gops[j];
                    rep_gops[j] = rep_gops[j+1];
                    rep_gops[j+1] = tmp;
                }

        double min_g  = rep_gops[0];
        double p5_g   = rep_gops[(int)(0.05 * N_REPS)];
        double med_g  = rep_gops[N_REPS / 2];
        double p95_g  = rep_gops[(int)(0.95 * N_REPS)];
        double max_g  = rep_gops[N_REPS - 1];
        double iqr_g  = rep_gops[(int)(0.75 * N_REPS)] - rep_gops[(int)(0.25 * N_REPS)];
        double mean_g = sum_gops / N_REPS;
        double sq_sum = 0.0;
        for (int r = 0; r < N_REPS; r++)
            sq_sum += (rep_gops[r] - mean_g) * (rep_gops[r] - mean_g);
        double std_g  = sqrt(sq_sum / N_REPS);
        double cv_g   = (med_g > 0) ? (std_g / med_g) * 100.0 : 0.0;
        double mxmd_g = (med_g > 0) ? (max_g / med_g) : 0.0;

        char layer_str[256];
        sprintf(layer_str, "%s (%dx%dx%d)", label, m, n, k);

        printf("  %-24s | %6.2f GOPs  "
               "[%5.2f, %5.2f, %5.2f, %5.2f, %5.2f | %5.3f, %5.3f, %3.0f%%, %4.2f]  %s (mc:%d nc:%d kc:%d)\n",
               layer_str,
               med_g, min_g, p5_g, med_g, p95_g, max_g,
               iqr_g, std_g, cv_g, mxmd_g,
               correct ? "PASS" : "FAIL", mc, nc, kc);

        my_aligned_free(A); my_aligned_free(B);
        my_aligned_free(C); my_aligned_free(Cg);
        my_aligned_free(Ac); my_aligned_free(Bc);
    }

    fclose(fp);
    freeMatrix(ukrmatrix);
    printf("\n");
    return 0;
}