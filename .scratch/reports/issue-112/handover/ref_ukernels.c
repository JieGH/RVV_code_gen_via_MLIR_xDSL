// Issue #112 hand-over: naive reference microkernels (see ref_ukernels.h). Plain loops, no vector code.
#include "ref_ukernels.h"

void ref_ukernel_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA,
                      int ldB, int Clda) {
    (void)ldA; (void)ldB;
    for (int j = 0; j < nr; j++)
        for (int i = 0; i < mr; i++) {
            float sum = beta != 0.0f ? Cr[j * Clda + i] : 0.0f;
            for (int p = 0; p < kc; p++) sum += Ar[p * mr + i] * Br[p * nr + j];
            Cr[j * Clda + i] = sum;
        }
}

void ref_ukernel_f16f16f32(int mr, int nr, int kc, const _Float16 *Ar, const _Float16 *Br, float *Cr, float beta,
                           int ldA, int ldB, int Clda) {
    (void)ldA; (void)ldB;
    for (int j = 0; j < nr; j++)
        for (int i = 0; i < mr; i++) {
            float sum = beta != 0.0f ? Cr[j * Clda + i] : 0.0f;
            for (int p = 0; p < kc; p++) sum += (float)Ar[p * mr + i] * (float)Br[p * nr + j];
            Cr[j * Clda + i] = sum;
        }
}
