#ifndef REF_UKERNELS_H
#define REF_UKERNELS_H
// Issue #112 hand-over: naive reference microkernels, for the correctness check only (no vector code).
// Same arguments as the xdsl_ukernel_* wrappers (and Héctor's uk_intrinsic_fp32 / uk_intrinsic_fp16), except that
// here mr and nr ARE used: they give the tile size (MR, NR), i.e. the packed strides of Ar and Br.
//   Ar[p*mr + i], Br[p*nr + j], Cr[j*Clda + i];  beta == 0: C = A*B, other beta: C += A*B.  Sums in FP32.
#ifdef __cplusplus
extern "C" {
#endif
void ref_ukernel_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA,
                      int ldB, int Clda);
void ref_ukernel_f16f16f32(int mr, int nr, int kc, const _Float16 *Ar, const _Float16 *Br, float *Cr, float beta,
                           int ldA, int ldB, int Clda);
#ifdef __cplusplus
}
#endif
#endif
