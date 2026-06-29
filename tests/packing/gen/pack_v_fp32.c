#include <riscv_vector.h>
#include <algorithm>

extern "C" {
void pack_CB_v_if_fp32_RR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 8;
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

void pack_CB_v_else_fp32_RR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 8;
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

void pack_RB_v_if_fp32_RR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
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

void pack_RB_v_else_fp32_RR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
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
