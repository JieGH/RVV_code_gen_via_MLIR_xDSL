#include "kernels_RVV_16x8_fp16.h"
#include <stdlib.h>
typedef void (*ukrFunction)( void *ctxt, int_fast32_t KC, const _Float16* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const _Float16* beta,  struct exo_win_2f16 C);
ukrFunction**** allocateMatrix();
void fillMatrix(ukrFunction**** matrix);
void freeMatrix(ukrFunction**** matrix);
