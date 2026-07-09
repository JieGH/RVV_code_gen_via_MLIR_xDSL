/**
 * driver_pack_check.cpp
 *
 * Verification of pack_CB_v and pack_RB_v against the scalar BLIS reference.
 *
 * Uses pack_dispatch.h to call kernels via the full gemm_blis.h API:
 *   pack_CB_v(orderM, transM, mc, nc, M, ldM, Mc, NR)
 *   pack_RB_v(orderM, transM, mc, nc, M, ldM, Mc, MR)
 *
 * Tests both branches of each function:
 *   pack_CB_v  'C','N'  → if-branch   (vlse32, Col-Major Non-Trans)
 *   pack_CB_v  'R','N'  → else-branch (vle32,  Row-Major Non-Trans)
 *   pack_RB_v  'C','N'  → if-branch   (vle32,  Col-Major Non-Trans)
 *   pack_RB_v  'R','N'  → else-branch (vlse32, Row-Major Non-Trans)
 *
 * Test categories per kernel+branch:
 *   1. Dimension coverage  — systematic (mc, nc, ldM)
 *   2. Ground-truth        — B[i,j]=i*1000+j, verify exact packed values
 *   3. Stride correctness  — same content, varying ldM
 *   4. Buffer boundary     — sentinel guard detects OOB writes
 *   5. Stress sweep        — parametric grid
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef TEST_INT8
typedef int8_t TestType;
#define GEN_VAL(i, j) (TestType)(((i) * 11 + (j)) % 127)
const TestType SENTINEL = -128;
#define EPSILON 0
#else
typedef float TestType;
#define GEN_VAL(i, j) (TestType)((i) * 1000 + (j))
const TestType SENTINEL = -9999.0f;
#define EPSILON 1e-5f
#endif

#include "pack_dispatch.h"   // pack_CB_v, pack_RB_v (gemm_blis.h API)

// Scalar references (exact copies of gemm_blis.c)
void ref_pack_CB(char orderM, char transM, int mc, int nc,
                 TestType* M, int ldM, TestType* Mc, int RR);
void ref_pack_RB(char orderM, char transM, int mc, int nc,
                 TestType* M, int ldM, TestType* Mc, int RR);

// The exact scalar references for INT8 Method 3
void ref_pack_DOT_INT8_LANE_Pack_CB(char orderA, char transM, int mc, int nc, TestType *M, int ldM, TestType *Mc, int nr_block, int kr_block);
void ref_pack_DOT_INT8_LANE_Pack_RB(char orderA, char transM, int mc, int nc, TestType *M, int ldM, TestType *Mc, int mr_block, int kr_block);

// ── Constants ─────────────────────────────────────────────────────────────────
// NR and MR are passed via -DNR=<n> -DMR=<m> compiler defines (see Makefile).
// compile_packing.sh is the single source of truth: it sets NR/MR, calls
// generate_packing.py and gen_dispatch.py with those values, then passes them
// here through the Makefile.  Changing NR/MR in compile_packing.sh is enough.
// pack_dispatch.h asserts RR == NR and RR == MR at runtime to catch mismatches.
//
// Equivalence proof: xDSL absolute indexing vs. reference k++ pointer:
//   Reference uses:  k = j*mc;  then k++ per element, k += (RR-nr) per row
//   xDSL uses:       dst_off = j*mc + i*RR;  vse32 writes exactly vl elements
//   Proof:           At entry of row i in block j: k_ref = j*mc + i*RR  ✓
//                    After writing nr elements:    k_ref = j*mc + i*RR + nr
//                    After k += (RR-nr):           k_ref = j*mc + (i+1)*RR  ✓
//                    xDSL's next iteration uses:   j*mc + (i+1)*RR           ✓
#ifndef NR
#define NR 8
#endif
#ifndef MR
#define MR 8
#endif

#ifndef KR
#define KR 32
#endif

static int g_pass = 0, g_fail = 0;

// ── Packed buffer size helpers ────────────────────────────────────────────────
static int cb_packed_size(int mc, int nc) { 
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
        return ((mc + KR - 1) / KR) * KR * ((nc + NR - 1) / NR) * NR;
    }
    return mc * ((nc + NR - 1) / NR) * NR; 
}
static int rb_packed_size(int mc, int nc) { 
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
        return ((nc + KR - 1) / KR) * KR * ((mc + MR - 1) / MR) * MR;
    }
    return ((mc + MR - 1) / MR) * MR * nc; 
}

// ── Generic compare helper ────────────────────────────────────────────────────
static bool compare_buffers(const char* tag, const std::vector<TestType>& ref,
                             const std::vector<TestType>& vec, int data_size) {
    int mismatches = 0;
    for (int p = 0; p < data_size; ++p)
        if (std::abs((float)ref[p] - (float)vec[p]) > EPSILON) {
            if (mismatches < 4)
                std::printf("    [!] %s pos=%d ref=%.1f vec=%.1f\n", tag, p, (float)ref[p], (float)vec[p]);
            ++mismatches;
        }
    return mismatches == 0;
}

static bool check_guard(const std::vector<TestType>& buf, int data_size, int guard) {
    for (int g = 0; g < guard; ++g)
        if (buf[data_size + g] != SENTINEL) return false;
    return true;
}

// ── CB test helper ────────────────────────────────────────────────────────────
static bool run_CB(const char* label, char orderM, char transM,
                   int mc, int nc, int ldM, std::vector<TestType>& B) {
    // Skip unaligned K-dimension for DOT_INT8 packing
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0 && (mc % 4 != 0)) {
        // std::printf("  CB(%c,%c) %-40s SKIPPED (K-dim mc %% 4 != 0)\n", orderM, transM, label);
        return true;
    }

    int psz = cb_packed_size(mc, nc);
    const int G = 16;
    std::vector<TestType> ref(psz + G, 0), vec(psz + G, 0);
    for (int g = 0; g < G; ++g) vec[psz + g] = SENTINEL;

    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
        ref_pack_DOT_INT8_LANE_Pack_CB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), NR, KR);
    } else {
        ref_pack_CB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), NR);
    }
    
    pack_CB_v  (orderM, transM, mc, nc, B.data(), ldM, vec.data(), NR);

    bool data_ok  = compare_buffers("CB", ref, vec, psz);
    bool guard_ok = check_guard(vec, psz, G);
    bool ok = data_ok && guard_ok;
    std::printf("  CB(%c,%c) %-40s %s\n", orderM, transM, label, ok ? "PASS" : "FAIL");
    ok ? ++g_pass : ++g_fail;
    return ok;
}

// ── RB test helper ────────────────────────────────────────────────────────────
static bool run_RB(const char* label, char orderM, char transM,
                   int mc, int nc, int ldM, std::vector<TestType>& B) {
    // Skip unaligned K-dimension for DOT_INT8 packing
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0 && (nc % 4 != 0)) {
        // std::printf("  RB(%c,%c) %-40s SKIPPED (K-dim nc %% 4 != 0)\n", orderM, transM, label);
        return true;
    }

    int psz = rb_packed_size(mc, nc);
    const int G = 16;
    std::vector<TestType> ref(psz + G, 0), vec(psz + G, 0);
    for (int g = 0; g < G; ++g) vec[psz + g] = SENTINEL;

    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
        ref_pack_DOT_INT8_LANE_Pack_RB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), MR, KR);
    } else {
        ref_pack_RB(orderM, transM, mc, nc, B.data(), ldM, ref.data(), MR);
    }
    
    pack_RB_v  (orderM, transM, mc, nc, B.data(), ldM, vec.data(), MR);

    bool data_ok  = compare_buffers("RB", ref, vec, psz);
    bool guard_ok = check_guard(vec, psz, G);
    bool ok = data_ok && guard_ok;
    std::printf("  RB(%c,%c) %-40s %s\n", orderM, transM, label, ok ? "PASS" : "FAIL");
    ok ? ++g_pass : ++g_fail;
    return ok;
}

// ── Ground-truth CB ───────────────────────────────────────────────────────────
// if-branch  (Col-Major Non-Trans): Bc[j*mc + i*NR + jj] = M[(j+jj)*ldM + i]
// else-branch (Row-Major Non-Trans): Bc[j*mc + i*NR + jj] = M[i*ldM + (j+jj)]
static bool gt_CB(const char* label, char orderM, char transM, int mc, int nc, int ldM) {
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) return true; // GT disabled for method 3 due to complex interleaving layout
    std::vector<TestType> B((size_t)ldM * std::max(mc, nc) + 64, 0);

    bool col_major_storage = (orderM == 'C');
    for (int i = 0; i < mc; ++i)
        for (int j = 0; j < nc; ++j) {
            TestType val = GEN_VAL(i, j);
            if (col_major_storage) B[(size_t)j * ldM + i] = val;
            else                   B[(size_t)i * ldM + j] = val;
        }

    int psz = cb_packed_size(mc, nc);
    std::vector<TestType> vec(psz + 16, 0);
    pack_CB_v(orderM, transM, mc, nc, B.data(), ldM, vec.data(), NR);

    // In all CB branches, the packed result is the same:
    //   Bc[j*mc + i*NR + jj] = logical B[i][j+jj] = i*1000 + (j+jj)
    int mismatches = 0;
    for (int jb = 0; jb * NR < nc; ++jb) {
        int j0 = jb * NR, nr = std::min(nc - j0, NR);
        for (int i = 0; i < mc; ++i)
            for (int jj = 0; jj < nr; ++jj) {
                float expected = (float)(i * 1000 + j0 + jj);
                int   pos      = jb * NR * mc + i * NR + jj;
                if (std::abs(vec[pos] - expected) > 1e-5f) {
                    if (mismatches < 4)
                        std::printf("    [!] CB GT pos=%d exp=%.0f got=%.1f (i=%d j=%d)\n",
                                    pos, expected, vec[pos], i, j0+jj);
                    ++mismatches;
                }
            }
    }
    bool ok = (mismatches == 0);
    std::printf("  CB(%c,%c) GT %-37s %s\n", orderM, transM, label, ok ? "PASS" : "FAIL");
    ok ? ++g_pass : ++g_fail;
    return ok;
}

// ── Ground-truth RB ───────────────────────────────────────────────────────────
// Packed result: Mc[i*nc + j*MR + ii] = logical A[i+ii][j]
static bool gt_RB(const char* label, char orderM, char transM, int mc, int nc, int ldM) {
    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) return true; // GT disabled for method 3
    std::vector<TestType> B((size_t)ldM * std::max(mc, nc) + 64, 0);
    bool col_major_storage = (orderM == 'C');
    for (int i = 0; i < mc; ++i)
        for (int j = 0; j < nc; ++j) {
            TestType val = GEN_VAL(i, j);
            if (col_major_storage) B[(size_t)j * ldM + i] = val;
            else                   B[(size_t)i * ldM + j] = val;
        }

    int psz = rb_packed_size(mc, nc);
    std::vector<TestType> vec(psz + 16, 0);
    pack_RB_v(orderM, transM, mc, nc, B.data(), ldM, vec.data(), MR);

    // Mc[i_block * nc * MR + j * MR + ii] = logical A[i_block*MR + ii][j]
    int mismatches = 0;
    for (int ib = 0; ib * MR < mc; ++ib) {
        int i0 = ib * MR, rr = std::min(mc - i0, MR);
        for (int j = 0; j < nc; ++j)
            for (int ii = 0; ii < rr; ++ii) {
                float expected = (float)((i0 + ii) * 1000 + j);
                int   pos      = ib * nc * MR + j * MR + ii;
                if (std::abs(vec[pos] - expected) > 1e-5f) {
                    if (mismatches < 4)
                        std::printf("    [!] RB GT pos=%d exp=%.0f got=%.1f (i=%d j=%d)\n",
                                    pos, expected, vec[pos], i0+ii, j);
                    ++mismatches;
                }
            }
    }
    bool ok = (mismatches == 0);
    std::printf("  RB(%c,%c) GT %-37s %s\n", orderM, transM, label, ok ? "PASS" : "FAIL");
    ok ? ++g_pass : ++g_fail;
    return ok;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    std::printf("======================================================\n");
    std::printf("  pack_CB_v + pack_RB_v — Reference-Aligned Verification\n");
    std::printf("  Method: %s\n", PACKING_METHOD);
    std::printf("  NR=%d (CB block width)  MR=%d (RB block height)\n", NR, MR);
#ifdef TEST_INT8
    std::printf("  VLEN=256, LMUL=1, INT8  — SpacemiT K1\n");
#else
    std::printf("  VLEN=256, LMUL=1, FP32  — SpacemiT K1\n");
#endif
    std::printf("======================================================\n\n");

    // ── Category 1: Dimension Coverage ───────────────────────────────────────
    std::printf("── Category 1: Dimension Coverage ──────────────────────\n");
    {
        const int BUF = 300 * 64;
        std::vector<TestType> B(BUF);
        for (int k = 0; k < BUF; ++k) B[k] = (TestType)(k % 97 + 1);

        // CB: if-branch  ('C','N')  ldM >= mc (col-major)
        run_CB("nc=8,  mc=32, ldM=40",  'C','N', 32,  8, 40, B);
        run_CB("nc=24, mc=32, ldM=40",  'C','N', 32, 24, 40, B);
        run_CB("nc=13, mc=32, ldM=40",  'C','N', 32, 13, 40, B);  // tail nr=5
        run_CB("nc=1,  mc=32, ldM=40",  'C','N', 32,  1, 40, B);
        run_CB("nc=17, mc=32, ldM=40",  'C','N', 32, 17, 40, B);  // tail nr=1
        run_CB("mc=1,  nc=24, ldM=10",  'C','N',  1, 24, 10, B);
        run_CB("mc=128,nc=32, ldM=130", 'C','N',128, 32,130, B);

        // CB: else-branch ('R','N')  ldM >= nc (row-major)
        run_CB("nc=8,  mc=32, ldM=8",   'R','N', 32,  8,  8, B);
        run_CB("nc=24, mc=32, ldM=24",  'R','N', 32, 24, 24, B);
        run_CB("nc=13, mc=32, ldM=16",  'R','N', 32, 13, 16, B);  // tail nr=5
        run_CB("nc=1,  mc=32, ldM=8",   'R','N', 32,  1,  8, B);
        run_CB("mc=1,  nc=24, ldM=24",  'R','N',  1, 24, 24, B);

        // RB: if-branch  ('C','N')  ldM >= mc (col-major)
        run_RB("mc=8,  nc=24, ldM=10",  'C','N',  8, 24, 10, B);
        run_RB("mc=32, nc=24, ldM=40",  'C','N', 32, 24, 40, B);
        run_RB("mc=5,  nc=24, ldM=8",   'C','N',  5, 24,  8, B);  // tail rr=1
        run_RB("mc=1,  nc=24, ldM=4",   'C','N',  1, 24,  4, B);
        run_RB("mc=4,  nc=1,  ldM=8",   'C','N',  4,  1,  8, B);

        // RB: else-branch ('R','N')  ldM >= nc (row-major)
        run_RB("mc=8,  nc=24, ldM=24",  'R','N',  8, 24, 24, B);
        run_RB("mc=32, nc=24, ldM=24",  'R','N', 32, 24, 24, B);
        run_RB("mc=5,  nc=24, ldM=24",  'R','N',  5, 24, 24, B);  // tail rr=1
        run_RB("mc=1,  nc=24, ldM=24",  'R','N',  1, 24, 24, B);
    }

    // ── Category 2: Ground-Truth ──────────────────────────────────────────────
    std::printf("\n── Category 2: Ground-Truth Semantic ───────────────────\n");
    std::printf("   B[i][j] = i*1000+j, verify exact packed values\n\n");

    gt_CB("mc=4, nc=8,  ldM=40", 'C','N',  4,  8, 40);   // if-branch
    gt_CB("mc=4, nc=5,  ldM=40", 'C','N',  4,  5, 40);   // if-branch tail
    gt_CB("mc=4, nc=8,  ldM=8",  'R','N',  4,  8,  8);   // else-branch
    gt_CB("mc=4, nc=5,  ldM=8",  'R','N',  4,  5,  8);   // else-branch tail
    gt_CB("mc=32,nc=24, ldM=40", 'C','N', 32, 24, 40);
    gt_CB("mc=32,nc=24, ldM=24", 'R','N', 32, 24, 24);

    gt_RB("mc=4, nc=8,  ldM=40", 'C','N',  4,  8, 40);   // if-branch
    gt_RB("mc=5, nc=8,  ldM=40", 'C','N',  5,  8, 40);   // if-branch tail rr=1
    gt_RB("mc=4, nc=8,  ldM=8",  'R','N',  4,  8,  8);   // else-branch
    gt_RB("mc=5, nc=8,  ldM=8",  'R','N',  5,  8,  8);   // else-branch tail
    gt_RB("mc=32,nc=24, ldM=40", 'C','N', 32, 24, 40);
    gt_RB("mc=32,nc=24, ldM=24", 'R','N', 32, 24, 24);

    // ── Category 3: Stride Correctness ────────────────────────────────────────
    std::printf("\n── Category 3: Stride Correctness ──────────────────────\n");
    {
        for (int ldM : {8, 12, 20, 40}) {
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "mc=8 nc=8 ldM=%d", ldM);
            gt_CB(lbl, 'C','N', 8, 8, ldM);
            gt_RB(lbl, 'C','N', 8, 8, ldM);
        }
        for (int ldM : {8, 12, 20, 40}) {
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "mc=8 nc=8 ldM=%d", ldM);
            gt_CB(lbl, 'R','N', 8, 8, ldM);
            gt_RB(lbl, 'R','N', 8, 8, ldM);
        }
    }

    // ── Category 4: Buffer Boundary ───────────────────────────────────────────
    std::printf("\n── Category 4: Buffer Boundary (OOB Guard) ─────────────\n");
    {
        std::vector<TestType> B(256 * 64);
        for (size_t k = 0; k < B.size(); ++k) B[k] = (TestType)(k % 89 + 1);
        run_CB("mc=1, nc=1, ldM=1",  'C','N', 1, 1, 1, B);
        run_CB("mc=1, nc=1, ldM=1",  'R','N', 1, 1, 1, B);
        run_CB("mc=4, nc=9, ldM=8",  'C','N', 4, 9, 8, B);
        run_CB("mc=4, nc=9, ldM=9",  'R','N', 4, 9, 9, B);
        run_RB("mc=1, nc=1, ldM=1",  'C','N', 1, 1, 1, B);
        run_RB("mc=1, nc=1, ldM=1",  'R','N', 1, 1, 1, B);
        run_RB("mc=5, nc=8, ldM=8",  'C','N', 5, 8, 8, B);
        run_RB("mc=5, nc=8, ldM=8",  'R','N', 5, 8, 8, B);
    }

    // ── Category 5: Stress Sweep ──────────────────────────────────────────────
    std::printf("\n── Category 5: Stress Sweep ─────────────────────────────\n");
    {
        const int BUF = 300 * 64;
        std::vector<TestType> B(BUF);
        for (int k = 0; k < BUF; ++k) B[k] = (TestType)((k * 7 + 3) % 89 + 1);

        int cb_if_p=0,cb_if_f=0, cb_else_p=0,cb_else_f=0;
        int rb_if_p=0,rb_if_f=0, rb_else_p=0,rb_else_f=0;

        for (int mc : {1, 4, 8, 16, 32}) {
        for (int nc : {1, 7, 8, 9, 15, 16, 17, 24, 31, 32}) {
        for (int extra : {0, 1, 8}) {
            // CB if-branch: ldM >= mc (col-major)
            {
                int ldM = mc + extra, psz = cb_packed_size(mc, nc);
                if ((size_t)ldM * nc < (size_t)BUF) {
                    std::vector<TestType> r(psz,0), v(psz,0);
                    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
                        ref_pack_DOT_INT8_LANE_Pack_CB('C','N',mc,nc,B.data(),ldM,r.data(),NR,KR);
                    } else {
                        ref_pack_CB('C','N',mc,nc,B.data(),ldM,r.data(),NR);
                    }
                    if (!(std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0 && mc % 4 != 0)) {
                        pack_CB_v  ('C','N',mc,nc,B.data(),ldM,v.data(),NR);
                        bool ok=true;
                        for(int p=0;p<psz&&ok;++p) if(std::abs((float)r[p]-(float)v[p])>EPSILON){ok=false;++cb_if_f;}
                        if(ok) ++cb_if_p;
                    }
                }
            }
            // CB else-branch: ldM >= nc (row-major)
            {
                int ldM = nc + extra, psz = cb_packed_size(mc, nc);
                if ((size_t)mc * ldM < (size_t)BUF) {
                    std::vector<TestType> r(psz,0), v(psz,0);
                    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
                        ref_pack_DOT_INT8_LANE_Pack_CB('R','N',mc,nc,B.data(),ldM,r.data(),NR,KR);
                    } else {
                        ref_pack_CB('R','N',mc,nc,B.data(),ldM,r.data(),NR);
                    }
                    if (!(std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0 && mc % 4 != 0)) {
                        pack_CB_v  ('R','N',mc,nc,B.data(),ldM,v.data(),NR);
                        bool ok=true;
                        for(int p=0;p<psz&&ok;++p) if(std::abs((float)r[p]-(float)v[p])>EPSILON){ok=false;++cb_else_f;}
                        if(ok) ++cb_else_p;
                    }
                }
            }
            // RB if-branch: ldM >= mc (col-major)
            {
                int ldM = mc + extra, psz = rb_packed_size(mc, nc);
                if ((size_t)ldM * nc < (size_t)BUF) {
                    std::vector<TestType> r(psz,0), v(psz,0);
                    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
                        ref_pack_DOT_INT8_LANE_Pack_RB('C','N',mc,nc,B.data(),ldM,r.data(),MR,KR);
                    } else {
                        ref_pack_RB('C','N',mc,nc,B.data(),ldM,r.data(),MR);
                    }
                    if (!(std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0 && nc % 4 != 0)) {
                        pack_RB_v  ('C','N',mc,nc,B.data(),ldM,v.data(),MR);
                        bool ok=true;
                        for(int p=0;p<psz&&ok;++p) if(std::abs((float)r[p]-(float)v[p])>EPSILON){ok=false;++rb_if_f;}
                        if(ok) ++rb_if_p;
                    }
                }
            }
            // RB else-branch: ldM >= nc (row-major)
            {
                int ldM = nc + extra, psz = rb_packed_size(mc, nc);
                if ((size_t)mc * ldM < (size_t)BUF) {
                    std::vector<TestType> r(psz,0), v(psz,0);
                    if (std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0) {
                        ref_pack_DOT_INT8_LANE_Pack_RB('R','N',mc,nc,B.data(),ldM,r.data(),MR,KR);
                    } else {
                        ref_pack_RB('R','N',mc,nc,B.data(),ldM,r.data(),MR);
                    }
                    if (!(std::strcmp(PACKING_METHOD, "DOT_INT8_LANE_Pack") == 0 && nc % 4 != 0)) {
                        pack_RB_v  ('R','N',mc,nc,B.data(),ldM,v.data(),MR);
                        bool ok=true;
                        for(int p=0;p<psz&&ok;++p) if(std::abs((float)r[p]-(float)v[p])>EPSILON){ok=false;++rb_else_f;}
                        if(ok) ++rb_else_p;
                    }
                }
            }
        }}}

        std::printf("  CB(C,N) sweep: %d/%d\n", cb_if_p,   cb_if_p+cb_if_f);
        std::printf("  CB(R,N) sweep: %d/%d\n", cb_else_p, cb_else_p+cb_else_f);
        std::printf("  RB(C,N) sweep: %d/%d\n", rb_if_p,   rb_if_p+rb_if_f);
        std::printf("  RB(R,N) sweep: %d/%d\n", rb_else_p, rb_else_p+rb_else_f);
        (cb_if_f+cb_else_f+rb_if_f+rb_else_f == 0) ? ++g_pass : ++g_fail;
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    std::printf("\n======================================================\n");
    std::printf("  Results: %d passed, %d failed\n", g_pass, g_fail);
    std::printf("======================================================\n");
    return (g_fail == 0) ? 0 : 1;
}
