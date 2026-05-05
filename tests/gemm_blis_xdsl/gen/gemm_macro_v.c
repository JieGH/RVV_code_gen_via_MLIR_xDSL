#include <riscv_vector.h>
#include <algorithm>
#include <stdint.h>

#define GET_PTR(ptr, off) (ptr + off)
#define GET_BETA(pc, beta) (pc == 0 ? beta : 1.0f)

extern "C" {
void gemm_microkernel_v(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac_ptr, float* Bc_ptr, float betaI, float* Cptr, int32_t ldC);

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

void pack_RB_v_if_fp32_RR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 8;
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

void pack_RB_v_else_fp32_RR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 8;
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

void gemm_blis_v_CN_CN_C_fp32(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13) {
  int32_t v14 = 0;
  int32_t v15 = 384;
  for (int32_t i16 = v14; i16 < v2; i16 += v15) {
    int32_t v17 = v2 - i16;
    int32_t v18 = 384;
    int32_t v19 = std::min(v17, v18);
    int32_t v20 = 256;
    for (int32_t j21 = v14; j21 < v3; j21 += v20) {
      int32_t v22 = v3 - j21;
      int32_t v23 = 256;
      int32_t v24 = std::min(v22, v23);
      int32_t v25 = i16 * v8;
      int32_t v26 = v25 + j21;
      float* v27 = GET_PTR(v7, v26);
      pack_CB_v_if_fp32_RR8(v27, v13, v24, v19, v8);
      float v28 = GET_BETA(j21, v9);
      int32_t v29 = 384;
      for (int32_t k30 = v14; k30 < v1; k30 += v29) {
        int32_t v31 = v1 - k30;
        int32_t v32 = 384;
        int32_t v33 = std::min(v31, v32);
        int32_t v34 = j21 * v6;
        int32_t v35 = v34 + k30;
        float* v36 = GET_PTR(v5, v35);
        pack_RB_v_if_fp32_RR8(v36, v12, v33, v24, v6);
        int32_t v37 = 8;
        for (int32_t l38 = v14; l38 < v19; l38 += v37) {
          int32_t v39 = v19 - l38;
          int32_t v40 = 8;
          int32_t v41 = std::min(v39, v40);
          int32_t v42 = 8;
          for (int32_t m43 = v14; m43 < v33; m43 += v42) {
            int32_t v44 = v33 - m43;
            int32_t v45 = 8;
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
            gemm_microkernel_v(v46, v41, v24, v4, v53, v55, v28, v51, v11);
          }
        }
      }
    }
  }
  return;
}



}
