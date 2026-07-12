#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <riscv_vector.h>

#define GET_PTR(ptr, off) (ptr + off)
#define GET_BETA(pc, beta) (pc == 0 ? beta : 1.0f)

extern "C" {
void pack_CB_v_if_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x1_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x2_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x3_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x4_b0(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x1_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x2_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x3_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x4_b1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);

void gemm_microkernel_v_dispatch_fp32_MR4_NR4(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (beta == 0.0f) {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_1x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_1x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_1x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_1x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_2x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_2x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_2x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_2x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_3x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_3x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_3x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_3x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_4x1_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_4x2_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_4x3_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_4x4_b0((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR=4 NR=4)\n", mr, nr); abort();
    } else {
        if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_1x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_1x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_1x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_1x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_2x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_2x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_2x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_2x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_3x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_3x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_3x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_3x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_4x1_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_4x2_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_4x3_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_4x4_b1((size_t)kc, &alpha, Ac, (size_t)4, Bc, (size_t)4, &beta, C, (size_t)ldc); return; }
        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR=4 NR=4)\n", mr, nr); abort();
    }
}

void gemm_microkernel_v_wrapper_fp32_MR4_NR4(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR4_NR4((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}
}

extern "C" {
void gemm_microkernel_v_fp32_MR4_NR4_1x1_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 1;
    float* v25 = &v3[v23];
    vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v24);
    size_t v27 = i22 * v6;
    vfloat32m1_t v28 = v21;
    float v29 = v5[v27];
    vfloat32m1_t v30 = __riscv_vfmacc_vf_f32m1(v28, v29, v26, v24);
    v21 = v30;
  }
  vfloat32m1_t v31 = v21;
  float* v32 = &v8[v11];
  __riscv_vse32_v_f32m1(v32, v31, v12);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_1x1_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v12);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 1;
    float* v26 = &v3[v24];
    vfloat32m1_t v27 = __riscv_vle32_v_f32m1(v26, v25);
    size_t v28 = i23 * v6;
    vfloat32m1_t v29 = v22;
    float v30 = v5[v28];
    vfloat32m1_t v31 = __riscv_vfmacc_vf_f32m1(v29, v30, v27, v25);
    v22 = v31;
  }
  vfloat32m1_t v32 = v22;
  float* v33 = &v8[v11];
  __riscv_vse32_v_f32m1(v33, v32, v12);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_1x2_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 1;
    float* v27 = &v3[v25];
    vfloat32m1_t v28 = __riscv_vle32_v_f32m1(v27, v26);
    size_t v29 = i24 * v6;
    size_t v30 = v29 + v12;
    vfloat32m1_t v31 = v22;
    float v32 = v5[v29];
    vfloat32m1_t v33 = __riscv_vfmacc_vf_f32m1(v31, v32, v28, v26);
    vfloat32m1_t v34 = v23;
    float v35 = v5[v30];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v28, v26);
    v22 = v33;
    v23 = v36;
  }
  vfloat32m1_t v37 = v22;
  vfloat32m1_t v38 = v23;
  float* v39 = &v8[v11];
  __riscv_vse32_v_f32m1(v39, v37, v12);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v12);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_1x2_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v12);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v12);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 1;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    vfloat32m1_t v33 = v24;
    float v34 = v5[v31];
    vfloat32m1_t v35 = __riscv_vfmacc_vf_f32m1(v33, v34, v30, v28);
    vfloat32m1_t v36 = v25;
    float v37 = v5[v32];
    vfloat32m1_t v38 = __riscv_vfmacc_vf_f32m1(v36, v37, v30, v28);
    v24 = v35;
    v25 = v38;
  }
  vfloat32m1_t v39 = v24;
  vfloat32m1_t v40 = v25;
  float* v41 = &v8[v11];
  __riscv_vse32_v_f32m1(v41, v39, v12);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v12);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_1x3_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 1;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v12);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v12);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v12);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_1x3_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v12);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v12);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v12);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 1;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v12);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v12);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v12);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_1x4_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 1;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v12);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v12);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v12);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v12);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_1x4_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v12);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v12);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v12);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v12);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 1;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v12);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v12);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v12);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v12);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_2x1_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 2;
    float* v25 = &v3[v23];
    vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v24);
    size_t v27 = i22 * v6;
    vfloat32m1_t v28 = v21;
    float v29 = v5[v27];
    vfloat32m1_t v30 = __riscv_vfmacc_vf_f32m1(v28, v29, v26, v24);
    v21 = v30;
  }
  vfloat32m1_t v31 = v21;
  float* v32 = &v8[v11];
  __riscv_vse32_v_f32m1(v32, v31, v13);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_2x1_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v13);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 2;
    float* v26 = &v3[v24];
    vfloat32m1_t v27 = __riscv_vle32_v_f32m1(v26, v25);
    size_t v28 = i23 * v6;
    vfloat32m1_t v29 = v22;
    float v30 = v5[v28];
    vfloat32m1_t v31 = __riscv_vfmacc_vf_f32m1(v29, v30, v27, v25);
    v22 = v31;
  }
  vfloat32m1_t v32 = v22;
  float* v33 = &v8[v11];
  __riscv_vse32_v_f32m1(v33, v32, v13);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_2x2_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 2;
    float* v27 = &v3[v25];
    vfloat32m1_t v28 = __riscv_vle32_v_f32m1(v27, v26);
    size_t v29 = i24 * v6;
    size_t v30 = v29 + v12;
    vfloat32m1_t v31 = v22;
    float v32 = v5[v29];
    vfloat32m1_t v33 = __riscv_vfmacc_vf_f32m1(v31, v32, v28, v26);
    vfloat32m1_t v34 = v23;
    float v35 = v5[v30];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v28, v26);
    v22 = v33;
    v23 = v36;
  }
  vfloat32m1_t v37 = v22;
  vfloat32m1_t v38 = v23;
  float* v39 = &v8[v11];
  __riscv_vse32_v_f32m1(v39, v37, v13);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v13);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_2x2_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v13);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v13);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 2;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    vfloat32m1_t v33 = v24;
    float v34 = v5[v31];
    vfloat32m1_t v35 = __riscv_vfmacc_vf_f32m1(v33, v34, v30, v28);
    vfloat32m1_t v36 = v25;
    float v37 = v5[v32];
    vfloat32m1_t v38 = __riscv_vfmacc_vf_f32m1(v36, v37, v30, v28);
    v24 = v35;
    v25 = v38;
  }
  vfloat32m1_t v39 = v24;
  vfloat32m1_t v40 = v25;
  float* v41 = &v8[v11];
  __riscv_vse32_v_f32m1(v41, v39, v13);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v13);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_2x3_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 2;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v13);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v13);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v13);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_2x3_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v13);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v13);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v13);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 2;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v13);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v13);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v13);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_2x4_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 2;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v13);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v13);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v13);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v13);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_2x4_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v13);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v13);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v13);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v13);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 2;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v13);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v13);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v13);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v13);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_3x1_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 3;
    float* v25 = &v3[v23];
    vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v24);
    size_t v27 = i22 * v6;
    vfloat32m1_t v28 = v21;
    float v29 = v5[v27];
    vfloat32m1_t v30 = __riscv_vfmacc_vf_f32m1(v28, v29, v26, v24);
    v21 = v30;
  }
  vfloat32m1_t v31 = v21;
  float* v32 = &v8[v11];
  __riscv_vse32_v_f32m1(v32, v31, v14);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_3x1_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 3;
    float* v26 = &v3[v24];
    vfloat32m1_t v27 = __riscv_vle32_v_f32m1(v26, v25);
    size_t v28 = i23 * v6;
    vfloat32m1_t v29 = v22;
    float v30 = v5[v28];
    vfloat32m1_t v31 = __riscv_vfmacc_vf_f32m1(v29, v30, v27, v25);
    v22 = v31;
  }
  vfloat32m1_t v32 = v22;
  float* v33 = &v8[v11];
  __riscv_vse32_v_f32m1(v33, v32, v14);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_3x2_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 3;
    float* v27 = &v3[v25];
    vfloat32m1_t v28 = __riscv_vle32_v_f32m1(v27, v26);
    size_t v29 = i24 * v6;
    size_t v30 = v29 + v12;
    vfloat32m1_t v31 = v22;
    float v32 = v5[v29];
    vfloat32m1_t v33 = __riscv_vfmacc_vf_f32m1(v31, v32, v28, v26);
    vfloat32m1_t v34 = v23;
    float v35 = v5[v30];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v28, v26);
    v22 = v33;
    v23 = v36;
  }
  vfloat32m1_t v37 = v22;
  vfloat32m1_t v38 = v23;
  float* v39 = &v8[v11];
  __riscv_vse32_v_f32m1(v39, v37, v14);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v14);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_3x2_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 3;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    vfloat32m1_t v33 = v24;
    float v34 = v5[v31];
    vfloat32m1_t v35 = __riscv_vfmacc_vf_f32m1(v33, v34, v30, v28);
    vfloat32m1_t v36 = v25;
    float v37 = v5[v32];
    vfloat32m1_t v38 = __riscv_vfmacc_vf_f32m1(v36, v37, v30, v28);
    v24 = v35;
    v25 = v38;
  }
  vfloat32m1_t v39 = v24;
  vfloat32m1_t v40 = v25;
  float* v41 = &v8[v11];
  __riscv_vse32_v_f32m1(v41, v39, v14);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v14);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_3x3_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 3;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v14);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v14);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v14);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_3x3_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v14);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 3;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v14);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v14);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v14);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_3x4_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 3;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v14);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v14);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v14);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v14);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_3x4_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v14);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v14);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 3;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v14);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v14);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v14);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v14);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_4x1_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 4;
    float* v25 = &v3[v23];
    vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v24);
    size_t v27 = i22 * v6;
    vfloat32m1_t v28 = v21;
    float v29 = v5[v27];
    vfloat32m1_t v30 = __riscv_vfmacc_vf_f32m1(v28, v29, v26, v24);
    v21 = v30;
  }
  vfloat32m1_t v31 = v21;
  float* v32 = &v8[v11];
  __riscv_vse32_v_f32m1(v32, v31, v15);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_4x1_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 4;
    float* v26 = &v3[v24];
    vfloat32m1_t v27 = __riscv_vle32_v_f32m1(v26, v25);
    size_t v28 = i23 * v6;
    vfloat32m1_t v29 = v22;
    float v30 = v5[v28];
    vfloat32m1_t v31 = __riscv_vfmacc_vf_f32m1(v29, v30, v27, v25);
    v22 = v31;
  }
  vfloat32m1_t v32 = v22;
  float* v33 = &v8[v11];
  __riscv_vse32_v_f32m1(v33, v32, v15);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_4x2_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 4;
    float* v27 = &v3[v25];
    vfloat32m1_t v28 = __riscv_vle32_v_f32m1(v27, v26);
    size_t v29 = i24 * v6;
    size_t v30 = v29 + v12;
    vfloat32m1_t v31 = v22;
    float v32 = v5[v29];
    vfloat32m1_t v33 = __riscv_vfmacc_vf_f32m1(v31, v32, v28, v26);
    vfloat32m1_t v34 = v23;
    float v35 = v5[v30];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v28, v26);
    v22 = v33;
    v23 = v36;
  }
  vfloat32m1_t v37 = v22;
  vfloat32m1_t v38 = v23;
  float* v39 = &v8[v11];
  __riscv_vse32_v_f32m1(v39, v37, v15);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v15);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_4x2_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 4;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    vfloat32m1_t v33 = v24;
    float v34 = v5[v31];
    vfloat32m1_t v35 = __riscv_vfmacc_vf_f32m1(v33, v34, v30, v28);
    vfloat32m1_t v36 = v25;
    float v37 = v5[v32];
    vfloat32m1_t v38 = __riscv_vfmacc_vf_f32m1(v36, v37, v30, v28);
    v24 = v35;
    v25 = v38;
  }
  vfloat32m1_t v39 = v24;
  vfloat32m1_t v40 = v25;
  float* v41 = &v8[v11];
  __riscv_vse32_v_f32m1(v41, v39, v15);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v15);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_4x3_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 4;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v15);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v15);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v15);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_4x3_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v15);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 4;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v15);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v15);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v15);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_4x4_b0(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 4;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v15);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v15);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v15);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v15);
  return;
}
void gemm_microkernel_v_fp32_MR4_NR4_4x4_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v15);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v15);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 4;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v15);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v15);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v15);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v15);
  return;
}
void gemm_macro_v_fp32_MR4_NR4(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
  int32_t v17 = 0;
  for (int32_t i18 = v17; i18 < v2; i18 += v15) {
    int32_t v19 = v2 - i18;
    int32_t v20 = std::min(v19, v15);
    for (int32_t j21 = v17; j21 < v3; j21 += v16) {
      int32_t v22 = v3 - j21;
      int32_t v23 = std::min(v22, v16);
      int32_t v24 = i18 * v8;
      int32_t v25 = v24 + j21;
      float* v26 = GET_PTR(v7, v25);
      pack_CB_v_if_fp32_MR4_NR4(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR4_NR4(v33, v12, v30, v23, v6);
        int32_t v34 = 4;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 4;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 4;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 4;
            int32_t v43 = std::min(v41, v42);
            int32_t v44 = k28 + m40;
            int32_t v45 = i18 + l35;
            int32_t v46 = v45 * v11;
            int32_t v47 = v46 + v44;
            float* v48 = GET_PTR(v10, v47);
            int32_t v49 = m40 * v23;
            float* v50 = GET_PTR(v12, v49);
            int32_t v51 = l35 * v23;
            float* v52 = GET_PTR(v13, v51);
            gemm_microkernel_v_dispatch_fp32_MR4_NR4(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void pack_CB_v_if_fp32_MR4_NR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = __riscv_vsetvl_e32m1(v12);
    size_t v14 = v5 * v8;
    for (size_t j15 = v6; j15 < v3; j15 += v7) {
      size_t v16 = i10 * v5;
      size_t v17 = v16 + j15;
      float* v18 = &v1[v17];
      vfloat32m1_t v19 = __riscv_vlse32_v_f32m1(v18, v14, v13);
      size_t v20 = i10 * v3;
      size_t v21 = j15 * v9;
      size_t v22 = v20 + v21;
      float* v23 = &v2[v22];
      __riscv_vse32_v_f32m1(v23, v19, v13);
    }
  }
  return;
}
void pack_CB_v_else_fp32_MR4_NR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    size_t v12 = __riscv_vsetvl_e32m1(v11);
    for (size_t j13 = v6; j13 < v3; j13 += v7) {
      size_t v14 = j13 * v5;
      size_t v15 = v14 + i9;
      float* v16 = &v1[v15];
      vfloat32m1_t v17 = __riscv_vle32_v_f32m1(v16, v12);
      size_t v18 = i9 * v3;
      size_t v19 = j13 * v8;
      size_t v20 = v18 + v19;
      float* v21 = &v2[v20];
      __riscv_vse32_v_f32m1(v21, v17, v12);
    }
  }
  return;
}
void pack_RB_v_if_fp32_MR4_NR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    size_t v12 = __riscv_vsetvl_e32m1(v11);
    for (size_t j13 = v6; j13 < v4; j13 += v7) {
      size_t v14 = j13 * v5;
      size_t v15 = v14 + i9;
      float* v16 = &v1[v15];
      vfloat32m1_t v17 = __riscv_vle32_v_f32m1(v16, v12);
      size_t v18 = i9 * v4;
      size_t v19 = j13 * v8;
      size_t v20 = v18 + v19;
      float* v21 = &v2[v20];
      __riscv_vse32_v_f32m1(v21, v17, v12);
    }
  }
  return;
}
void pack_RB_v_else_fp32_MR4_NR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = __riscv_vsetvl_e32m1(v12);
    size_t v14 = v5 * v8;
    for (size_t j15 = v6; j15 < v4; j15 += v7) {
      size_t v16 = i10 * v5;
      size_t v17 = v16 + j15;
      float* v18 = &v1[v17];
      vfloat32m1_t v19 = __riscv_vlse32_v_f32m1(v18, v14, v13);
      size_t v20 = i10 * v4;
      size_t v21 = j15 * v9;
      size_t v22 = v20 + v21;
      float* v23 = &v2[v22];
      __riscv_vse32_v_f32m1(v23, v19, v13);
    }
  }
  return;
}


}
