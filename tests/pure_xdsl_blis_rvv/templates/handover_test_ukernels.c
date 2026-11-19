// Issue #112: check every hand-over microkernel against the naive reference (ref_ukernels.c), both called
// through Héctor's own function-pointer types.
// Generated from tests/pure_xdsl_blis_rvv/templates/handover_test_ukernels.c by gen_handover_ukernels.py.
//
// Per kernel: kc in {1, 7, 256}; beta in {0, 1, 0.5}; two data passes:
//   exact  : whole numbers -5..5 (exact in FP16, products and sums exact in FP32): result must match bit for bit.
//   random : values in [0, 1) (rounded to FP16 for the mixed kernels): ||C - ref|| / ||ref|| <= max(1e-5, kc * 1e-6),
//            the error and tolerance of our GEMM driver (templates/driver_benchmark.cpp).
// Reference: ref_ukernel_fp32 / ref_ukernel_f16f16f32 on the same inputs. beta != 0 means C += A*B, as in his kernels.
// C is column-major with Clda = MR + 3: the 3 extra rows of each column must stay untouched.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xdsl_ukernels_fp32.h"
#include "xdsl_ukernels_f16f16f32.h"
#include "ref_ukernels.h"

// Copied from GEMMLib_yolo b292c92, src/GEMM/gemm_blis.h: assigning our wrappers to these checks the signature.
typedef _Float16 fp16_t;
typedef void (*uk_intrinsic_fp32)(int, int, int, const float *, const float *, float *, float, int, int, int);
typedef void (*uk_intrinsic_fp16)(int, int, int, const fp16_t *, const fp16_t *, float *, float, int, int, int);

struct entry_fp32 { int mr, nr; uk_intrinsic_fp32 fn; };
struct entry_f16f16f32 { int mr, nr; uk_intrinsic_fp16 fn; };

static const uk_intrinsic_fp32 ref_fp32 = ref_ukernel_fp32;
static const uk_intrinsic_fp16 ref_f16 = ref_ukernel_f16f16f32;

/*@TABLES@*/

#define NTAB(t) ((int)(sizeof(t) / sizeof((t)[0])))
#define PAD 3
#define SENTINEL 777.0f

static float val(int exact) { return exact ? (float)(rand() % 11 - 5) : (float)rand() / (float)RAND_MAX; }

// Returns the number of wrong elements for one call.
static int check_one(uk_intrinsic_fp32 f32, uk_intrinsic_fp16 f16, int mr, int nr, int kc, float beta, int exact) {
    int lda = mr + PAD;
    float *af = malloc(sizeof(float) * mr * kc), *bf = malloc(sizeof(float) * nr * kc);
    fp16_t *ah = malloc(sizeof(fp16_t) * mr * kc), *bh = malloc(sizeof(fp16_t) * nr * kc);
    float *c = malloc(sizeof(float) * lda * nr), *cref = malloc(sizeof(float) * lda * nr);
    for (int i = 0; i < mr * kc; i++) { af[i] = val(exact); ah[i] = (fp16_t)af[i]; if (f16) af[i] = (float)ah[i]; }
    for (int i = 0; i < nr * kc; i++) { bf[i] = val(exact); bh[i] = (fp16_t)bf[i]; if (f16) bf[i] = (float)bh[i]; }
    for (int j = 0; j < nr; j++)
        for (int i = 0; i < lda; i++) cref[j * lda + i] = i < mr ? val(exact) : SENTINEL;
    memcpy(c, cref, sizeof(float) * lda * nr);

    if (f16) { f16(mr, nr, kc, ah, bh, c, beta, 0, 0, lda); ref_f16(mr, nr, kc, ah, bh, cref, beta, 0, 0, lda); }
    else     { f32(mr, nr, kc, af, bf, c, beta, 0, 0, lda); ref_fp32(mr, nr, kc, af, bf, cref, beta, 0, 0, lda); }

    double tol = fmax(1e-5, kc * 1e-6), err2 = 0.0, nrm2 = 0.0;
    int bad = 0;
    for (int j = 0; j < nr; j++)
        for (int i = 0; i < lda; i++) {
            float got = c[j * lda + i];
            if (i >= mr) { bad += got != SENTINEL; continue; }
            double ref = cref[j * lda + i];
            err2 += (got - ref) * (got - ref);
            nrm2 += ref * ref;
            int wrong = exact && got != cref[j * lda + i];
            if (wrong && bad < 3)
                fprintf(stderr, "  %s %dx%d kc=%d beta=%g %s: C(%d,%d) = %.9g, ref %.9g\n", f16 ? "f16f16f32" : "fp32",
                        mr, nr, kc, beta, exact ? "exact" : "random", i, j, got, ref);
            bad += wrong;
        }
    double rel = nrm2 > 0.0 ? sqrt(err2 / nrm2) : sqrt(err2);
    if (!exact && !(rel <= tol)) {
        fprintf(stderr, "  %s %dx%d kc=%d beta=%g random: relative error %.3g > %.3g\n", f16 ? "f16f16f32" : "fp32", mr,
                nr, kc, beta, rel, tol);
        bad++;
    }
    free(af); free(bf); free(ah); free(bh); free(c); free(cref);
    return bad;
}

int main(void) {
    static const int kcs[] = {1, 7, 256};
    static const float betas[] = {0.0f, 1.0f, 0.5f};
    int total_bad = 0;
    srand(112);
    for (int dt = 0; dt < 2; dt++) {
        int n = dt ? NTAB(table_f16f16f32) : NTAB(table_fp32), calls = 0, bad_kernels = 0;
        for (int t = 0; t < n; t++) {
            int mr = dt ? table_f16f16f32[t].mr : table_fp32[t].mr, nr = dt ? table_f16f16f32[t].nr : table_fp32[t].nr;
            int bad = 0;
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    for (int exact = 1; exact >= 0; exact--, calls++)
                        bad += check_one(dt ? NULL : table_fp32[t].fn, dt ? table_f16f16f32[t].fn : NULL, mr, nr, kcs[a],
                                         betas[b], exact);
            if (bad) { printf("FAIL %s %dx%d: %d wrong elements\n", dt ? "f16f16f32" : "fp32", mr, nr, bad); bad_kernels++; }
            total_bad += bad;
        }
        printf("%-9s : %d kernels, %d calls, %d kernels failed -> %s\n", dt ? "f16f16f32" : "fp32", n, calls, bad_kernels,
               bad_kernels ? "FAIL" : "PASS");
    }
    return total_bad ? 1 : 0;
}
