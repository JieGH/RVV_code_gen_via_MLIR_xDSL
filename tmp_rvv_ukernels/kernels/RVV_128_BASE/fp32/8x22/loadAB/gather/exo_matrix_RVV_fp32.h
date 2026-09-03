#include "kernels_RVV_8x22_fp32.h"
#include <stdlib.h>
typedef void (*ukrFunction)( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta,  struct exo_win_2f32 C);
ukrFunction**** allocateMatrix();
void fillMatrix(ukrFunction**** matrix);
void freeMatrix(ukrFunction**** matrix);
