#pragma once
#include <stddef.h>
#include <stdint.h>
#include "kernels_RVV_4x4_i8i8i32.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ukrFunction)(void* ctxt, int kc, float* alpha, int8_t* A, int lda, int8_t* B, int ldb, float* beta, int32_t* C, int ldc);

ukrFunction**** allocateMatrix();
void fillMatrix(ukrFunction**** matrix);
void freeMatrix(ukrFunction**** matrix);

#ifdef __cplusplus
}
#endif
