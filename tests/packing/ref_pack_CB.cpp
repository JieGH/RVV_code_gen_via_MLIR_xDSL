/**
 * ref_pack_CB.cpp
 *
 * Scalar reference implementations copied verbatim from:
 *   .ext/RVV_ukernels_benchmark/gemm_blis_family/src/gemm_blis.c
 *
 * Used as ground-truth correctness reference for the vectorized
 * pack_CB_v and pack_RB_v kernels.
 *
 * Macro: Mcol(i,j) = M[j*ldM + i]   (Column-Major indexing)
 */

#include <algorithm>

#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

// ---------------------------------------------------------------------------
// pack_CB — exact copy from gemm_blis.c line 203-235
//
// Packs B matrix into Column-Block (CB) packed layout with block width RR=NR.
//
// if  branch: Non-Transposed Col-Major  OR  Transposed Row-Major
//   Mc[k] = Mcol(i, j+jj) = M[(j+jj)*ldM + i]   ← stride = ldM (non-unit)
//
// else branch: Non-Transposed Row-Major  OR  Transposed Col-Major
//   Mc[k] = Mcol(j+jj, i) = M[i*ldM + (j+jj)]   ← stride = 1   (unit)
// ---------------------------------------------------------------------------
void ref_pack_CB( char orderM, char transM, int mc, int nc,
                  float *M, int ldM, float *Mc, int RR ) {
    int i, j, jj, k, nr;

    k = 0;
    if ( ((transM=='N')&&(orderM=='C')) ||
         ((transM=='T')&&(orderM=='R')) )
        for ( j=0; j<nc; j+=RR ) {
            k = j*mc;
            nr = std::min( nc-j, RR );
            for ( i=0; i<mc; i++ ) {
                for ( jj=0; jj<nr; jj++ ) {
                    Mc[k] = Mcol(i, j+jj);
                    k++;
                }
                k += (RR-nr);
            }
        }
    else
        for ( j=0; j<nc; j+=RR ) {
            k = j*mc;
            nr = std::min( nc-j, RR );
            for ( i=0; i<mc; i++ ) {
                for ( jj=0; jj<nr; jj++ ) {
                    Mc[k] = Mcol(j+jj, i);
                    k++;
                }
                k += (RR-nr);
            }
        }
}

// ---------------------------------------------------------------------------
// pack_RB — exact copy from gemm_blis.c line 168-201
//
// Packs A matrix into Row-Block (RB) packed layout with block height RR=MR.
//
// Outer loop: row blocks (i step MR)
// Inner loop: all columns (j step 1)
//
// if  branch: Non-Transposed Col-Major  OR  Transposed Row-Major
//   Mc[k] = Mcol(i+ii, j) = M[j*ldM + (i+ii)]   ← for fixed j, stride = 1 (unit)
//
// else branch: Non-Transposed Row-Major  OR  Transposed Col-Major
//   Mc[k] = Mcol(j, i+ii) = M[(i+ii)*ldM + j]   ← for fixed j, stride = ldM (non-unit)
// ---------------------------------------------------------------------------
void ref_pack_RB( char orderM, char transM, int mc, int nc,
                  float *M, int ldM, float *Mc, int RR ) {
    int i, j, ii, k, rr;

    if ( ((transM=='N')&&(orderM=='C')) ||
         ((transM=='T')&&(orderM=='R')) ) {
        for ( i=0; i<mc; i+=RR ) {
            k = i*nc;
            rr = std::min( mc-i, RR );
            for ( j=0; j<nc; j++ ) {
                for ( ii=0; ii<rr; ii++ ) {
                    Mc[k] = Mcol(i+ii, j);
                    k++;
                }
                k += (RR-rr);
            }
        }
    } else {
        for ( i=0; i<mc; i+=RR ) {
            k = i*nc;
            rr = std::min( mc-i, RR );
            for ( j=0; j<nc; j++ ) {
                for ( ii=0; ii<rr; ii++ ) {
                    Mc[k] = Mcol(j, i+ii);
                    k++;
                }
                k += (RR-rr);
            }
        }
    }
}
