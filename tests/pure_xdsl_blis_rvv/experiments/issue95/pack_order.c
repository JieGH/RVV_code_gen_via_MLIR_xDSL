// Issue #95: time packing a column-major A (m x k, lda = m) into MR-row panels, in two read orders.
//   panel-outer (ours, gemm_pack_xdsl.cpp pack_RB_v_*): for each MR-row panel, for each column: read MR floats.
//   column-outer (Héctor's pack_generic_MRxkc_A):       for each column, for each MR-row panel: read MR floats.
// Same bytes read and written, same packed layout (checked). Blocks of mc rows, as the GEMM packs them.
// Build on the board: gcc -O2 -march=rv64gcv pack_order.c -o pack_order
// Run: taskset -c 4 ./pack_order M K MR MC KC REPS   ->  ms per full pack of A, in MC x KC blocks as the GEMM packs it
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static void copy_mr(const float *src, float *dst, int mr) {  // RVV copy of mr floats, 8 per vector (VLEN 256)
    for (int x = 0; x < mr; ) { size_t vl = __riscv_vsetvl_e32m1(mr - x); __riscv_vse32_v_f32m1(dst + x, __riscv_vle32_v_f32m1(src + x, vl), vl); x += vl; }
}

static void pack_panel_outer(const float *A, int lda, int mc, int kc, float *Ac, int MR) {
    for (int i = 0; i < mc; i += MR)
        for (int j = 0; j < kc; j++) copy_mr(&A[(size_t)j * lda + i], &Ac[(size_t)i * kc + (size_t)j * MR], mc - i < MR ? mc - i : MR);
}

static void pack_column_outer(const float *A, int lda, int mc, int kc, float *Ac, int MR) {
    for (int j = 0; j < kc; j++)
        for (int i = 0; i < mc; i += MR) copy_mr(&A[(size_t)j * lda + i], &Ac[(size_t)i * kc + (size_t)j * MR], mc - i < MR ? mc - i : MR);
}

typedef void (*pack_fn)(const float *, int, int, int, float *, int);

static double time_pack(pack_fn f, const float *A, int m, int k, int MR, int MC, int KC, float *Ac, int reps) {
    double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int pc = 0; pc < k; pc += KC)  // GEMM order: pc outside ic; last blocks may be short
            for (int ic = 0; ic < m; ic += MC) f(&A[(size_t)pc * m + ic], m, m - ic < MC ? m - ic : MC, k - pc < KC ? k - pc : KC, Ac, MR);
    return (now() - t0) / reps * 1e3;
}

int main(int argc, char **argv) {
    if (argc != 7) { fprintf(stderr, "usage: %s M K MR MC KC REPS\n", argv[0]); return 2; }
    int m = atoi(argv[1]), k = atoi(argv[2]), MR = atoi(argv[3]), MC = atoi(argv[4]), KC = atoi(argv[5]), reps = atoi(argv[6]);
    if (KC > k) KC = k;
    if (MC > m) MC = m;
    size_t acn = (size_t)((MC + MR - 1) / MR * MR) * KC;  // whole panels: a short last panel still sits MR*k apart
    float *A = aligned_alloc(64, sizeof(float) * (size_t)m * k), *A1 = calloc(acn, sizeof(float)), *A2 = calloc(acn, sizeof(float));
    for (size_t x = 0; x < (size_t)m * k; x++) A[x] = (float)(x % 1013);
    pack_panel_outer(A, m, MC, KC, A1, MR); pack_column_outer(A, m, MC, KC, A2, MR);  // same layout?
    if (memcmp(A1, A2, sizeof(float) * acn)) { fprintf(stderr, "layouts differ\n"); return 1; }
    time_pack(pack_panel_outer, A, m, k, MR, MC, KC, A1, 2); time_pack(pack_column_outer, A, m, k, MR, MC, KC, A2, 2);  // warm
    double tp = time_pack(pack_panel_outer, A, m, k, MR, MC, KC, A1, reps);
    double tc = time_pack(pack_column_outer, A, m, k, MR, MC, KC, A2, reps);
    printf("A %dx%d MR %d MC %d KC %d reps %d | panel-outer (ours) %.3f ms | column-outer (his) %.3f ms\n", m, k, MR, MC, KC, reps, tp, tc);
    return 0;
}
