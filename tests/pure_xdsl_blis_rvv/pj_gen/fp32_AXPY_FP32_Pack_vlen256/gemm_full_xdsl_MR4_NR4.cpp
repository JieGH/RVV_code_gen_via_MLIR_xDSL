#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <riscv_vector.h>

#define GET_PTR(ptr, off) (ptr + off)
#define GET_BETA(pc, beta) (pc == 0 ? beta : 1.0f)

extern "C" {
void pack_CB_v_if_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void pack_CB_v_else_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_if_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void pack_RB_v_else_fp32_MR4_NR4(float*, float*, size_t, size_t, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x2(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x3(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_1x4(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x2(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x3(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_2x4(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x2(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x3(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_3x4(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x1(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x2(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x3(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);
void gemm_microkernel_v_fp32_MR4_NR4_4x4(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);

void gemm_microkernel_v_dispatch_fp32_MR4_NR4(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {
    if (mr == 1 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_1x1((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 1 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_1x2((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 1 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_1x3((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 1 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_1x4((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 2 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_2x1((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 2 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_2x2((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 2 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_2x3((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 2 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_2x4((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 3 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_3x1((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 3 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_3x2((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 3 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_3x3((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 3 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_3x4((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 4 && nr == 1) { gemm_microkernel_v_fp32_MR4_NR4_4x1((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 4 && nr == 2) { gemm_microkernel_v_fp32_MR4_NR4_4x2((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 4 && nr == 3) { gemm_microkernel_v_fp32_MR4_NR4_4x3((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
    if (mr == 4 && nr == 4) { gemm_microkernel_v_fp32_MR4_NR4_4x4((size_t)kc, &alpha, Ac, (size_t)mr, Bc, (size_t)nr, &beta, C, (size_t)ldc); return; }
}

void gemm_microkernel_v_wrapper_fp32_MR4_NR4(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {
    gemm_microkernel_v_dispatch_fp32_MR4_NR4((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);
}
}

extern "C" {
void gemm_microkernel_v_fp32_MR4_NR4_1x1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_1x2(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_1x3(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_1x4(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_2x1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_2x2(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_2x3(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_2x4(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_3x1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_3x2(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_3x3(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_3x4(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_4x1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_4x2(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_4x3(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_microkernel_v_fp32_MR4_NR4_4x4(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
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
void gemm_macro_v_fp32_MR4_NR4(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13) {
  int32_t v14 = 0;
  int32_t v15 = 1365;
  for (int32_t i16 = v14; i16 < v2; i16 += v15) {
    int32_t v17 = v2 - i16;
    int32_t v18 = 1365;
    int32_t v19 = std::min(v17, v18);
    int32_t v20 = 512;
    for (int32_t j21 = v14; j21 < v3; j21 += v20) {
      int32_t v22 = v3 - j21;
      int32_t v23 = 512;
      int32_t v24 = std::min(v22, v23);
      int32_t v25 = i16 * v8;
      int32_t v26 = v25 + j21;
      float* v27 = GET_PTR(v7, v26);
      pack_CB_v_if_fp32_MR4_NR4(v27, v13, v24, v19, v8);
      float v28 = GET_BETA(j21, v9);
      int32_t v29 = 192;
      for (int32_t k30 = v14; k30 < v1; k30 += v29) {
        int32_t v31 = v1 - k30;
        int32_t v32 = 192;
        int32_t v33 = std::min(v31, v32);
        int32_t v34 = j21 * v6;
        int32_t v35 = v34 + k30;
        float* v36 = GET_PTR(v5, v35);
        pack_RB_v_if_fp32_MR4_NR4(v36, v12, v33, v24, v6);
        int32_t v37 = 4;
        for (int32_t l38 = v14; l38 < v19; l38 += v37) {
          int32_t v39 = v19 - l38;
          int32_t v40 = 4;
          int32_t v41 = std::min(v39, v40);
          int32_t v42 = 4;
          for (int32_t m43 = v14; m43 < v33; m43 += v42) {
            int32_t v44 = v33 - m43;
            int32_t v45 = 4;
            int32_t v46 = std::min(v44, v45);
            int32_t v47 = k30 + m43;
            int32_t v48 = i16 + l38;
            int32_t v49 = v48 * v11;
            int32_t v50 = v49 + v47;
            float* v51 = GET_PTR(v10, v50);
            int32_t v52 = m43 * v24;
            float* v53 = GET_PTR(v12, v52);
            int32_t v54 = l38 * v24;
            float* v55 = GET_PTR(v13, v54);
            gemm_microkernel_v_dispatch_fp32_MR4_NR4(v46, v41, v24, v4, v53, v55, v28, v51, v11);
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
