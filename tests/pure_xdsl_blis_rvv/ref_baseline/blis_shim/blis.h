/* blis.h — minimal stand-in for the real BLIS header.
 *
 * WHY THIS EXISTS
 * ---------------
 * ext/RVV_ukernels_benchmark/gemm_blis_family unconditionally does
 *   #include "blis.h"                      (src/gemm_blis.h:43, src/test_gemm.c:44)
 * and src/test_gemm.c:87-92 calls bli_init() / bli_gks_query_cntx() /
 * bli_cntx_get_l3_sup_ker_dt() inside a `#if !defined(BLIS)` block, so a full
 * BLIS install would otherwise be a hard build dependency on the RISC-V board.
 *
 * WHEN THIS IS SAFE
 * -----------------
 * ONLY for RUN_MODE=FAMILY_EXO. In that mode the BLIS micro-kernel is never
 * invoked: gemm_blis.c:117-133 dispatches through `ukrmatrix[mr][nr][be]` (the
 * Exo kernel table) and the `gemm_kernel(...)` call at gemm_blis.c:139 sits in
 * the #else branch that FAMILY_EXO compiles out. `cntx` and `aux` are only
 * passed down as opaque pointers and never dereferenced.
 *
 * So the shim changes nothing that is measured. run_ref_baseline.sh refuses to
 * build with the shim unless RUN_MODE=FAMILY_EXO, and prints a warning either
 * way. If you ever want FAMILY_BLIS numbers, install real BLIS and pass
 * --blis-home instead of --use-shim.
 */
#ifndef BLIS_SHIM_H
#define BLIS_SHIM_H

#include <stddef.h>

typedef enum { BLIS_FLOAT = 0, BLIS_DOUBLE = 1 } num_t;
typedef enum { BLIS_GEMM_UKR = 0 } ukr_t;

typedef struct cntx_s    { int unused; } cntx_t;
typedef struct auxinfo_s { int unused; } auxinfo_t;

/* Signature matches BLIS's gemm micro-kernel type; never called under FAMILY_EXO. */
typedef void (*gemm_ukr_ft)( long m, long n, long k,
                             void *alpha, void *a, void *b,
                             void *beta,  void *c, long rs_c, long cs_c,
                             auxinfo_t *data, const cntx_t *cntx );

static inline void bli_init( void ) { }

static inline const cntx_t *bli_gks_query_cntx( void ) {
    static cntx_t dummy_cntx;
    return &dummy_cntx;
}

static inline gemm_ukr_ft bli_cntx_get_l3_sup_ker_dt( num_t dt, ukr_t ukr,
                                                      const cntx_t *cntx ) {
    (void)dt; (void)ukr; (void)cntx;
    return (gemm_ukr_ft)0;   /* unused under FAMILY_EXO */
}

static inline gemm_ukr_ft bli_cntx_get_l3_vir_ukr_dt( num_t dt, ukr_t ukr,
                                                      const cntx_t *cntx ) {
    return bli_cntx_get_l3_sup_ker_dt( dt, ukr, cntx );
}

#endif /* BLIS_SHIM_H */
