#ifndef XDSL_UKERNELS_FP32_H
#define XDSL_UKERNELS_FP32_H
// Issue #112 hand-over. Arguments as Héctor's uk_intrinsic_fp32 / uk_intrinsic_fp16 (GEMMLib_yolo b292c92).
// A: packed MR per k step; B: packed NR per k step; C column-major, Cr[j*Clda + i].
// mr, nr, ldA, ldB are not used: the full MR x NR tile is always computed.
// beta == 0: C = A*B. Any other beta: C += A*B (beta's value is not used). No alpha.
// _Float16 is the same type as his fp16_t.
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void xdsl_ukernel_8x1_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x2_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x3_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x4_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x5_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x6_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x7_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x8_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x9_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x10_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x11_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x12_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x13_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x14_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x15_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_8x16_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x1_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x2_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x3_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x4_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x5_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x6_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x7_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x8_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x9_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x10_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x11_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x12_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x13_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x14_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x15_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_16x16_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x1_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x2_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x3_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x4_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x5_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x6_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x7_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x8_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x9_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x10_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x11_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x12_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x13_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x14_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x15_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x16_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x1_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x2_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x3_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x4_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x5_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x6_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x7_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x8_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x9_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x10_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x11_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x12_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x13_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x14_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x15_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_32x16_fp32(int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
#ifdef __cplusplus
}
#endif
#endif // XDSL_UKERNELS_FP32_H
