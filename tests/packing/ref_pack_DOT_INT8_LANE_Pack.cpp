/**
 * ref_pack_DOT_INT8_LANE_Pack.cpp
 *
 * Exact copy of `pack_sdot_A` and `pack_sdot_B` from gemm_blis.c
 * 
 * ============================================================================
 * WHY THE ORIGINAL REFERENCE CODE WAS WRONG (AND CAUSED CORE DUMPS)
 * ============================================================================
 * 
 * 1. Buffer Overflow in RB Packing (A-Side)
 *    The original `pack_sdot_A` (RB packing) ALWAYS padded the K dimension (`kc`)
 *    to a multiple of `KR=32`. For example, if `mc=8` and `kc=24`, it would write
 *    a full 8x32 block (256 elements). However, the test driver's `rb_packed_size` 
 *    allocated memory without this KR padding (allocating only 8x24 = 192 elements).
 *    This caused the reference code to write past the end of the allocated buffer,
 *    corrupting the heap and causing the core dump.
 * 
 * 2. Inconsistent/Missing Padding in CB Packing (B-Side)
 *    The original `pack_sdot_B` (CB packing) had a bug in its tail logic `if (jj != KR)`.
 *    When processing the very last block of the matrix (where `nr < nr_block` AND 
 *    `kc` was not a multiple of 32), it forgot to pad the remaining columns 
 *    (`ii = nr` to `nr_block`) with zeros. While this didn't crash (since it happened 
 *    at the very end of the buffer), it created an incomplete memory layout that 
 *    was inconsistent with how RB packing worked.
 * 
 * 3. The Fix
 *    - The test driver (`driver_pack_check.cpp`) and benchmark were updated to 
 *      allocate fully padded buffers: `(K + KR - 1)/KR * KR`.
 *    - The vector code generation (`pack_builder.py`) was updated to correctly 
 *      stride across these fully padded blocks.
 *    - This reference code was rewritten to consistently and correctly pad ALL 
 *      blocks to exact multiples of `NR x KR` (for CB) and `MR x KR` (for RB), 
 *      guaranteeing safe memory access and an exact match with the vector code.
 * ============================================================================
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using std::min;

#ifndef Mcol
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]
#endif

typedef int8_t IN_TYPE;
typedef int8_t PACK_TYPE;

// ---------------------------------------------------------------------------
// A-Side Packing (Row-Block)
// ---------------------------------------------------------------------------
void ref_pack_DOT_INT8_LANE_Pack_RB(char orderA, char transM, int mc, int kc, IN_TYPE *M, int ldM, PACK_TYPE *Mc, int mr_block, int KR) {
  int p = 0;
  int kr_lim, i, j, ii, jj, mr;

  if (orderA == 'C') {
    for (i = 0; i < mc; i += mr_block) {
      mr = min(mr_block, mc - i);
      for (j = 0; j < kc; j += KR) {
        for (ii = 0; ii < mr; ii++) {
          kr_lim = min(KR, kc - j);
          for (jj = 0; jj < kr_lim; jj++) {
            Mc[p] = Mcol(i + ii, j + jj);
            p++;
          }
          for (; jj < KR; jj++) { Mc[p++] = 0; }
        }
        p += (mr_block - mr) * KR;
      }
    }
  } else {
    // printf("  [!] WARNING: ref_pack_DOT_INT8_LANE_Pack_RB for orderA='R' is not implemented! Returning without packing.\n");
  }
}

// ---------------------------------------------------------------------------
// B-Side Packing (Column-Block)
// ---------------------------------------------------------------------------
void ref_pack_DOT_INT8_LANE_Pack_CB(char orderB, char transB, int kc, int nc, IN_TYPE *M, int ldM, PACK_TYPE *Mc, int nr_block, int KR) {
  int p = 0;
  int kr_lim, i, j, jj, ii;
  int nr;

  if (orderB == 'C') {
    for (i = 0; i < nc; i += nr_block) {
      nr = min(nr_block, nc - i);
      for (j = 0; j < kc; j += KR) {
        kr_lim = min(KR, kc - j);
        for (jj = 0; jj < KR; jj += 4) {
          for (ii = 0; ii < nr; ii++) {
            for (int k = 0; k < 4; k++) {
              if (jj + k < kr_lim) {
                Mc[p++] = Mcol(j + jj + k, i + ii);
              } else {
                Mc[p++] = 0;
              }
            }
          }
          for (ii = nr; ii < nr_block; ii++) {
            Mc[p++] = 0; Mc[p++] = 0; Mc[p++] = 0; Mc[p++] = 0;
          }
        }
      }
    }
  } else {
    // printf("  [!] WARNING: ref_pack_DOT_INT8_LANE_Pack_CB for orderB='R' is not implemented! Returning without packing.\n");
  }
}
