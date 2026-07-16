/*
 * ============================================================================
 * SCALAR BASELINE: DOT-PRODUCT STYLE MATMUL FOR FP32
 * ============================================================================
 *
 * This file establishes the scalar (no SIMD intrinsics) reference design
 * for a DOT-product oriented GEMM, mirroring the structure and data layouts
 * used in the INT8/NEON microkernel (uk_intrinsic_quantize_int8_4x16_sdot)
 * and its packing routines (pack_sdot_A / pack_sdot_B) from GEMMLib, but
 * adapted for FP32 data types.
 *
 * PURPOSE
 * -------
 * This file serves as the C reference design for later code generation via
 * xDSL/MLIR. The scalar loops here map 1:1 to the MLIR IR that the code
 * generator should produce.
 *
 * RELATIONSHIP TO THE NEON INTRINSIC CODE (transformerIME/GEMMLib)
 * -----------------------------------------------------------------
 * The INT8 microkernel uses the NEON instruction:
 *
 *   vdotq_laneq_s32(C_acc, B_vec16, A_vec16, lane)
 *
 * which computes, for each of 4 output lanes i = 0..3:
 *
 *   C_acc[i] += B_vec16[i*4+0] * A_vec16[lane*4+0]
 *             + B_vec16[i*4+1] * A_vec16[lane*4+1]
 *             + B_vec16[i*4+2] * A_vec16[lane*4+2]
 *             + B_vec16[i*4+3] * A_vec16[lane*4+3]
 *
 * The scalar code below replaces that with an explicit 4-element inner loop.
 *
 * RELATIONSHIP TO THE PAPER (arxiv 2506.11728v1)
 * ------------------------------------------------
 * Section 5.2 (ARMv8.2 NEON, Fig. 4) describes exactly this microkernel.
 * The micro-tile is MR x NR = 4 x 16. KR = 16 for INT8; KR = 4 here
 * (one "DOT lane" = 4 FP32 multiplications, matching the 4-element DOT
 * granularity of vdotq_laneq_s32).
 *
 * BLOCKING / TILE DIMENSIONS
 * ---------------------------
 *   MR  = 4    rows  of A per micro-kernel call
 *   NR  = 16   cols  of B per micro-kernel call
 *   KR  = 4    elements per "DOT lane" (inner-most accumulation depth)
 *
 *   Macro-kernel blocking:
 *   MC = multiple of MR  (fits in L2 cache for Ac)
 *   NC = multiple of NR  (fits in L3 cache for Bc)
 *   KC = multiple of KR  (fits in L1 cache for a micro-panel)
 *
 * DATA LAYOUTS IN PACKED BUFFERS
 * --------------------------------
 *
 *  Packed A (Ar):  logical shape  [mc/MR][kc/KR][MR][KR]
 *
 *    For each [MR x KR] block, elements stored ROW-FIRST:
 *      Ar[i*KR + l]  = A(row_i, k+l)   for i=0..MR-1, l=0..KR-1
 *
 *    Full tile at offset  (tile_i * kc + tile_k*KR*MR):
 *      rows: tile_i*MR .. tile_i*MR + MR - 1
 *      cols: tile_k*KR .. tile_k*KR + KR - 1
 *
 *  Packed B (Br):  logical shape  [nc/NR][kc/KR][NR][KR]
 *
 *    For each [KR x NR] block, elements stored COLUMN-FIRST:
 *      Br[j*KR + l]  = B(k+l, col_j)   for j=0..NR-1, l=0..KR-1
 *
 *    For each output column j, all KR depth elements are stored consecutively.
 *    This matches pack_sdot_B() exactly: the innermost write is
 *      for each col ii: B(k+0,col), B(k+1,col), B(k+2,col), B(k+3,col)
 *    which is a stride-1 DOT-product group that vdotq_laneq_s32 reads as
 *    4 consecutive int8 values from the same column.
 *
 * COMPILE & RUN (standalone validation):
 *   gcc -O2 -DSCALAR_BASELINE_MAIN scalar_dot_gemm_fp32.c -lm -o baseline
 *   ./baseline
 *
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Tile parameters ---- */
#define MR   4    /* micro-tile rows                */
#define NR   16   /* micro-tile cols                */
#define KR   4    /* DOT lane depth                 */

/* ---- Column-major indexing macros ---- */
#define Mcol(r,c)  M[(c)*(ldM) + (r)]
#define Ccol(r,c)  Cr[(c)*(ldC) + (r)]


/* ===========================================================================
 * pack_A_fp32
 *
 * Packs an (mc x kc) column-major sub-block of A into the flat buffer Ar.
 *
 * Output layout: [mc/MR][kc/KR][MR * KR]
 *   Within each [MR x KR] tile: Ar[i*KR + l] = A(row_i, col_l)
 *
 * Corresponds to:  pack_sdot_A() in GEMMLib/src/gemm/gemm_blis.c
 *
 * NEON equivalent (INT8, KR=16 per call):
 *   A0 = vld1q_s8(&Ar[0]);    // row 0, 16 int8 elements
 *   A1 = vld1q_s8(&Ar[16]);   // row 1
 *   A2 = vld1q_s8(&Ar[32]);   // row 2
 *   A3 = vld1q_s8(&Ar[48]);   // row 3
 *   => here each row has KR=4 floats instead
 *
 * Args:
 *   mc   – rows to pack
 *   kc   – K depth to pack
 *   M    – ptr to A(ic, pc) in original matrix
 *   ldM  – column stride of A (= m for col-major)
 *   Ar   – output flat buffer (mc * kc floats)
 *   MR_  – register tile row count (= MR)
 * =========================================================================== */
void pack_A_fp32(int mc, int kc,
                 const float *M, int ldM,
                 float *Ar, int MR_)
{
    int p = 0;

    for (int i = 0; i < mc; i += MR_) {
        int mr = (mc - i < MR_) ? (mc - i) : MR_;

        for (int j = 0; j < kc; j += KR) {
            int kr = (kc - j < KR) ? (kc - j) : KR;

            /* Store MR rows, each with KR depth values (row-first within tile) */
            for (int ii = 0; ii < mr; ii++) {
                for (int jj = 0; jj < kr; jj++)
                    Ar[p++] = Mcol(i + ii, j + jj);   /* A(row, k_depth) */
                for (int jj = kr; jj < KR; jj++)
                    Ar[p++] = 0.0f;                    /* zero-pad K edge */
            }
            for (int ii = mr; ii < MR_; ii++)
                for (int jj = 0; jj < KR; jj++)
                    Ar[p++] = 0.0f;                    /* zero-pad row edge */
        }
    }
}


/* ===========================================================================
 * pack_B_fp32
 *
 * Packs a (kc x nc) column-major sub-block of B into the flat buffer Br.
 *
 * Output layout: [nc/NR][kc/KR][NR * KR]  (COLUMN-FIRST within each tile)
 *   Within each [NR x KR] tile:
 *     Br[j*KR + l] = B(k+l, col_j)   for j=0..NR-1, l=0..KR-1
 *
 * Corresponds EXACTLY to: pack_sdot_B() in GEMMLib/src/gemm/gemm_blis.c
 *
 * Reference inner loop (jj=0, one group of 4 in pack_sdot_B):
 *   for ii in 0..nr:           // col loop is OUTER
 *     Mc[p++] = B(j+0, i+ii)  // KR consecutive k-depth values per col
 *     Mc[p++] = B(j+1, i+ii)
 *     Mc[p++] = B(j+2, i+ii)
 *     Mc[p++] = B(j+3, i+ii)
 *
 * NEON interpretation: B0 = vld1q_s8(&Br[0]) loads 16 int8:
 *   B0[0..3]  = B(k+0..3, col0)  <- 4-element DOT group for col 0
 *   B0[4..7]  = B(k+0..3, col1)  <- col 1
 *   B0[8..11] = B(k+0..3, col2)  <- col 2
 *   B0[12..15]= B(k+0..3, col3)  <- col 3
 * vdotq_laneq_s32(C00, B0, A0, lane) computes 4 columns simultaneously.
 *
 * FP32 scalar equivalent: Br_ptr[j*KR + l] accessed as Br_ptr[col*KR + depth]
 *
 * Args:
 *   kc   – K dimension (rows of B)
 *   nc   – N dimension (cols of B)
 *   M    – ptr to B(pc, jc) in original matrix
 *   ldM  – column stride of B (= k for col-major)
 *   Br   – output flat buffer (kc * nc floats)
 *   NR_  – register tile col count (= NR)
 * =========================================================================== */
void pack_B_fp32(int kc, int nc,
                 const float *M, int ldM,
                 float *Br, int NR_)
{
    int p = 0;

    for (int j = 0; j < nc; j += NR_) {
        int nr = (nc - j < NR_) ? (nc - j) : NR_;

        for (int i = 0; i < kc; i += KR) {
            int kr = (kc - i < KR) ? (kc - i) : KR;

            /*
             * COLUMN-FIRST: for each output column ii, store KR depth values.
             * Matches pack_sdot_B's inner loop exactly:
             *   for (ii=0; ii<nr; ii++):
             *     Mc[p++] = B(k+0, col_ii)
             *     Mc[p++] = B(k+1, col_ii)
             *     Mc[p++] = B(k+2, col_ii)
             *     Mc[p++] = B(k+3, col_ii)
             */
            for (int ii = 0; ii < nr; ii++) {
                for (int jj = 0; jj < kr; jj++)
                    Br[p++] = Mcol(i + jj, j + ii);   /* B(k+jj, col_ii) */
                for (int jj = kr; jj < KR; jj++)
                    Br[p++] = 0.0f;                    /* zero-pad K edge */
            }
            /* zero-pad missing output columns to a full NR_ slot */
            for (int ii = nr; ii < NR_; ii++)
                for (int jj = 0; jj < KR; jj++)
                    Br[p++] = 0.0f;
        }
    }
}


/* ===========================================================================
 * microkernel_fp32_dot
 *
 * Computes:  Cr = beta*Cr + Ar * Br   for an (mr x nr) output tile.
 *
 * This is the scalar equivalent of uk_intrinsic_quantize_int8_4x16_sdot()
 * in GEMMLib/src/uk_quantize/quk.c, adapted for FP32.
 *
 * ALGORITHM (mirrors paper arxiv:2506.11728v1 Section 5.2, Figure 4):
 *
 *   1. Load or zero-init MR x NR accumulator tile C_tile.
 *   2. Outer K loop (stride KR): one iteration = one "vdotq" group.
 *      a. Load MR rows of Ar (each KR floats) -> A_row[MR][KR]
 *      b. For l = 0..KR-1  (the "lane" loop, unrolled in NEON as Rep+0..3):
 *           Load B_lane = &Br[l*NR]          (NR floats)
 *           For i = 0..MR-1:
 *             a_val = A_row[i][l]            (scalar broadcast)
 *             For j = 0..NR-1:
 *               C_tile[i][j] += a_val * B_lane[j]   <- dot contribution
 *   3. Write back C_tile to Cr.
 *
 * NEON MAPPING TABLE (INT8 -> FP32 scalar):
 * ┌──────────────────────────────────┬──────────────────────────────────────┐
 * │ NEON (INT8, quk.c)               │ Scalar FP32 (this function)          │
 * ├──────────────────────────────────┼──────────────────────────────────────┤
 * │ A0=vld1q_s8(&Ar[0])              │ A_row[0][0..3] = Ar_ptr[0..3]        │
 * │ A1=vld1q_s8(&Ar[16])             │ A_row[1][0..3] = Ar_ptr[4..7]        │
 * │ A2=vld1q_s8(&Ar[32])             │ A_row[2][0..3] = Ar_ptr[8..11]       │
 * │ A3=vld1q_s8(&Ar[48])             │ A_row[3][0..3] = Ar_ptr[12..15]      │
 * ├──────────────────────────────────┼──────────────────────────────────────┤
 * │ B0=vld1q_s8(&Br[0])   [Rep+0]     │ Br_ptr[j*KR+l] for j=0..NR-1,l=0    │
 * │ B0[0..3]  =B(k+0..3, col0)        │ Br_ptr[0*KR+0..3] = B(k+0..3,col0)  │
 * │ B0[4..7]  =B(k+0..3, col1)        │ Br_ptr[1*KR+0..3] = B(k+0..3,col1)  │
 * ├──────────────────────────────────┼──────────────────────────────────────┤
 * │ C00=vdotq_laneq_s32(C00,B0,A0,0) │ for j: C_tile[0][j] +=               │
 * │  computes 4 cols simultaneously   │   A_row[0][l] * Br_ptr[j*KR+l]      │
 * └──────────────────────────────────┴──────────────────────────────────────┘
 *
 * Args:
 *   mr, nr – actual tile size (≤ MR, NR; border tiles may be smaller)
 *   kc     – K depth
 *   Ar     – packed A micro-panel, layout [kc/KR][MR*KR]
 *   Br     – packed B micro-panel, layout [kc/KR][KR*NR]
 *   Cr     – pointer to C(ic+ir, jc+jr), column-major
 *   beta   – 0.0 = write, 1.0 = accumulate
 *   ldC    – leading dim of C (column stride)
 * =========================================================================== */
void microkernel_fp32_dot(int mr, int nr, int kc,
                          const float *Ar, const float *Br,
                          float *Cr, float beta, int ldC)
{
    /* Local MR x NR accumulator tile (row-major for cache efficiency) */
    float C_tile[MR * NR];

    /* 1. Initialize accumulator */
    if (beta == 0.0f) {
        memset(C_tile, 0, sizeof(C_tile));
    } else {
        for (int i = 0; i < MR; i++)
            for (int j = 0; j < NR; j++)
                C_tile[i * NR + j] = (i < mr && j < nr)
                                     ? beta * Ccol(i, j) : 0.0f;
    }

    const float *Ar_ptr = Ar;
    const float *Br_ptr = Br;

    /* 2. K outer loop: stride KR  <==>  one iteration of L6 in paper */
    for (int k = 0; k < kc; k += KR) {

        /*
         * Load A: MR rows x KR elements each.
         * Ar_ptr layout: [row i=0..MR-1][depth l=0..KR-1]
         *   => Ar_ptr[i*KR + l]
         *
         * NEON: A0=vld1q_s8(&Ar[0]); A1=vld1q_s8(&Ar[16]); ...
         *   (16 int8 per row; here KR=4 floats per row)
         */
        float A_row[MR][KR];
        for (int i = 0; i < MR; i++)
            for (int l = 0; l < KR; l++)
                A_row[i][l] = Ar_ptr[i * KR + l];

        /*
         * Lane loop: l = 0..KR-1
         * Each iteration = one "Rep" block in quk.c
         * (Rep+0 => l=0, Rep+1 => l=1, Rep+2 => l=2, Rep+3 => l=3)
         *
         * In NEON (INT8), the "lane" argument to vdotq_laneq_s32
         * selects which 4-element chunk of A to broadcast.
         * Here we just index A_row[i][l].
         */
        for (int l = 0; l < KR; l++) {
            /*
             * B access: Br_ptr layout is COLUMN-FIRST [col j][depth l]
             *   => Br_ptr[j * KR + l]  for col j, depth l
             *
             * NEON mapping: B0 = vld1q_s8(&Br[0]) loads 16 int8:
             *   B0[0..3]   = B(k+0..3, col0)  ← Br_ptr[0*KR + 0..3]
             *   B0[4..7]   = B(k+0..3, col1)  ← Br_ptr[1*KR + 0..3]
             *   B0[8..11]  = B(k+0..3, col2)  ← Br_ptr[2*KR + 0..3]
             *   B0[12..15] = B(k+0..3, col3)  ← Br_ptr[3*KR + 0..3]
             * vdotq_laneq_s32 then computes a 4-element dot for 4 cols at once.
             * Here we compute all NR cols with scalar multiply-accumulate.
             */
            for (int i = 0; i < MR; i++) {
                float a_val = A_row[i][l];   /* broadcast element from A row i */
                for (int j = 0; j < NR; j++)
                    C_tile[i * NR + j] += a_val * Br_ptr[j * KR + l];
                    /*                                ^^^^^^^^^^^^
                     * Column-first B access: col j, depth l
                     * Matches Br[j*KR+l] = B(k+l, col_j) from pack_B_fp32
                     */
            }
        }

        Ar_ptr += MR * KR;   /* advance A panel pointer by one KR block */
        Br_ptr += KR * NR;   /* advance B panel pointer by one KR block */
    }

    /* 3. Write back only the actual mr x nr region to C (column-major) */
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            Ccol(i, j) = C_tile[i * NR + j];
}


/* ===========================================================================
 * gemm_fp32_dot_scalar
 *
 * BLIS-style 5-loop GEMM macro-kernel:  C = A*B + beta*C
 *
 * Loop structure (mirrors gemm_B3A2C0 in GEMMLib/src/gemm/gemm_blis.c):
 *   L1: jc [0..n) step NC                -- N tile
 *   L2: pc [0..k) step KC                -- K tile  => pack B into Bc
 *   L3: ic [0..m) step MC                -- M tile  => pack A into Ac
 *   L4: jr [0..nc) step NR               \
 *   L5: ir [0..mc) step MR               /  => microkernel
 *
 * Assumes: column-major A, B, C; no transposition.
 *
 * Args:
 *   m, n, k       – matrix dimensions
 *   A, ldA        – input matrix A (m x k, col-major)
 *   B, ldB        – input matrix B (k x n, col-major)
 *   beta, C, ldC  – output: C = A*B + beta*C
 *   MC, NC, KC    – macro-kernel blocking parameters
 * =========================================================================== */
void gemm_fp32_dot_scalar(int m, int n, int k,
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
    if (!Ac || !Bc) { free(Ac); free(Bc); return; }

    float one = 1.0f, betaI;

    /* L1: tile over N */
    for (int jc = 0; jc < n; jc += NC) {
        int nc = (n - jc < NC) ? (n - jc) : NC;

        /* L2: tile over K */
        for (int pc = 0; pc < k; pc += KC) {
            int kc = (k - pc < KC) ? (k - pc) : KC;
            int kc_pad = ((kc + KR - 1) / KR) * KR;

            /* Pack B(pc:pc+kc, jc:jc+nc) into Bc */
            pack_B_fp32(kc, nc,
                        &B[pc + (long)jc * ldB],
                        ldB, Bc, NR);

            betaI = (pc == 0) ? beta : one;

            /* L3: tile over M */
            for (int ic = 0; ic < m; ic += MC) {
                int mc = (m - ic < MC) ? (m - ic) : MC;

                /* Pack A(ic:ic+mc, pc:pc+kc) into Ac */
                pack_A_fp32(mc, kc,
                            &A[ic + (long)pc * ldA],
                            ldA, Ac, MR);

                /* L4 + L5: micro-kernel tiles */
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = (nc - jr < NR) ? (nc - jr) : NR;
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = (mc - ir < MR) ? (mc - ir) : MR;

                        /* A micro-panel: Ac[ir*kc_pad .. (ir+MR)*kc_pad) */
                        const float *Ar = &Ac[(long)ir * kc_pad];
                        /* B micro-panel: Bc[jr*kc_pad .. (jr+NR)*kc_pad) */
                        const float *Br = &Bc[(long)jr * kc_pad];
                        /* C output tile: C(ic+ir, jc+jr), col-major */
                        float *Cr = &C[(ic + ir) + (long)(jc + jr) * ldC];

                        microkernel_fp32_dot(mr, nr, kc_pad,
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


/* ===========================================================================
 * gemm_fp32_naive  –  naive triple-loop reference (column-major)
 * =========================================================================== */
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
                sum += A[i + (long)p * ldA] * B[p + (long)j * ldB];
            C[i + (long)j * ldC] = beta * C[i + (long)j * ldC] + sum;
        }
}


/* ===========================================================================
 * VALIDATION DRIVER
 *
 * Build:  gcc -O2 -DSCALAR_BASELINE_MAIN scalar_dot_gemm_fp32.c -lm -o baseline
 * Run:    ./baseline
 * =========================================================================== */
#ifdef SCALAR_BASELINE_MAIN

int main(void)
{
    srand(42);

    /* Dimensions that exercise both full tiles and border (partial) tiles */
    int m = 12, n = 20, k = 16;
    int ldA = m, ldB = k, ldC = m;   /* column-major */

    float *A  = malloc(m * k * sizeof(float));
    float *B  = malloc(k * n * sizeof(float));
    float *C1 = malloc(m * n * sizeof(float));  /* naive result    */
    float *C2 = malloc(m * n * sizeof(float));  /* dot-style result */

    for (int i = 0; i < m*k; i++) A[i]  = (float)(rand()%200 - 100) / 20.f;
    for (int i = 0; i < k*n; i++) B[i]  = (float)(rand()%200 - 100) / 20.f;
    for (int i = 0; i < m*n; i++) C1[i] = C2[i] = (float)(rand()%100) / 10.f;

    float beta = 0.5f;

    gemm_fp32_naive(m, n, k, A, ldA, B, ldB, beta, C1, ldC);
    gemm_fp32_dot_scalar(m, n, k, A, ldA, B, ldB, beta, C2, ldC,
                         m, n, k);   /* single macro-tile for test */

    double max_err = 0.0;
    int err_count = 0;
    for (int i = 0; i < m*n; i++) {
        double e = fabs((double)C1[i] - (double)C2[i]);
        if (e > 1e-4 && err_count < 5) {
            printf("Mismatch at %d: naive=%f, dot=%f, diff=%f\n", i, C1[i], C2[i], e);
            err_count++;
        }
        if (e > max_err) max_err = e;
    }

    printf("Max absolute error vs naive: %.3e  ->  %s\n",
           max_err, (max_err < 1e-4) ? "PASS" : "FAIL");

    free(A); free(B); free(C1); free(C2);
    return (max_err < 1e-4) ? 0 : 1;
}
#endif /* SCALAR_BASELINE_MAIN */
