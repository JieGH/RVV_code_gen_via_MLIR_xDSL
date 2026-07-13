#include "gemm_headers.h"

extern "C" {

void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR1(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 1;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v3; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v3;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR1(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 1;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v3; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v3;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR1(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 1;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v4; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v4;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR1(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 1;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v4; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v4;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR2(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 2;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v3; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v3;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR2(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 2;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v3; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v3;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR2(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 2;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v4; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v4;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR2(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 2;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v4; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v4;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR3(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 3;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v3; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v3;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR3(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 3;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v3; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v3;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR3(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 3;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v4; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v4;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR3(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 3;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v4; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v4;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v3; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v3;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v3; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v3;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v4; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v4;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v4; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v4;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR5(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 5;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v3; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v3;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR5(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 5;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v3; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v3;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR5(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 5;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v4; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v4;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR5(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 5;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v4; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v4;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR6(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 6;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v3; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v3;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR6(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 6;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v3; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v3;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR6(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 6;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v4; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v4;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR6(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 6;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v4; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v4;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR7(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 7;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v3; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v3;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR7(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 7;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v3; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v3;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR7(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 7;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v4; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v4;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR7(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 7;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v4; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v4;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 8;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v3; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v3;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 8;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v3; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v3;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 8;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v4; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v4;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
    }
  }
  return;
}
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 8;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v4; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v4;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
    }
  }
  return;
}
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR9(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 9;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v3; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v3;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
      size_t v32 = 8;
      size_t v33 = 16;
      size_t v34 = std::min(v12, v33);
      size_t v35 = std::max(v34, v32);
      size_t v36 = v35 - v32;
      size_t v37 = __riscv_vsetvl_e32m1(v36);
      size_t v38 = v32 * v5;
      size_t v39 = v16 + v38;
      float* v40 = &v1[v39];
      vfloat32m1_t v41 = __riscv_vlse32_v_f32m1(v40, v13, v37);
      size_t v42 = v19 + v32;
      float* v43 = &v2[v42];
      __riscv_vse32_v_f32m1(v43, v41, v37);
    }
  }
  return;
}
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR9(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 9;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v3; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v3;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
      size_t v29 = 8;
      size_t v30 = 16;
      size_t v31 = std::min(v11, v30);
      size_t v32 = std::max(v31, v29);
      size_t v33 = v32 - v29;
      size_t v34 = __riscv_vsetvl_e32m1(v33);
      size_t v35 = v14 + v29;
      float* v36 = &v1[v35];
      vfloat32m1_t v37 = __riscv_vle32_v_f32m1(v36, v34);
      size_t v38 = v17 + v29;
      float* v39 = &v2[v38];
      __riscv_vse32_v_f32m1(v39, v37, v34);
    }
  }
  return;
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR9(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 9;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v4; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v4;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
      size_t v29 = 8;
      size_t v30 = 16;
      size_t v31 = std::min(v11, v30);
      size_t v32 = std::max(v31, v29);
      size_t v33 = v32 - v29;
      size_t v34 = __riscv_vsetvl_e32m1(v33);
      size_t v35 = v14 + v29;
      float* v36 = &v1[v35];
      vfloat32m1_t v37 = __riscv_vle32_v_f32m1(v36, v34);
      size_t v38 = v17 + v29;
      float* v39 = &v2[v38];
      __riscv_vse32_v_f32m1(v39, v37, v34);
    }
  }
  return;
}
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR9(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 9;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v4; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v4;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
      size_t v32 = 8;
      size_t v33 = 16;
      size_t v34 = std::min(v12, v33);
      size_t v35 = std::max(v34, v32);
      size_t v36 = v35 - v32;
      size_t v37 = __riscv_vsetvl_e32m1(v36);
      size_t v38 = v32 * v5;
      size_t v39 = v16 + v38;
      float* v40 = &v1[v39];
      vfloat32m1_t v41 = __riscv_vlse32_v_f32m1(v40, v13, v37);
      size_t v42 = v19 + v32;
      float* v43 = &v2[v42];
      __riscv_vse32_v_f32m1(v43, v41, v37);
    }
  }
  return;
}
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR10(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 10;
  for (size_t i10 = v6; i10 < v4; i10 += v9) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v3; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v3;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
      size_t v32 = 8;
      size_t v33 = 16;
      size_t v34 = std::min(v12, v33);
      size_t v35 = std::max(v34, v32);
      size_t v36 = v35 - v32;
      size_t v37 = __riscv_vsetvl_e32m1(v36);
      size_t v38 = v32 * v5;
      size_t v39 = v16 + v38;
      float* v40 = &v1[v39];
      vfloat32m1_t v41 = __riscv_vlse32_v_f32m1(v40, v13, v37);
      size_t v42 = v19 + v32;
      float* v43 = &v2[v42];
      __riscv_vse32_v_f32m1(v43, v41, v37);
    }
  }
  return;
}
void pack_CB_v_else_AXPY_FP32_Pack_fp32_NR10(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 10;
  for (size_t i9 = v6; i9 < v4; i9 += v8) {
    size_t v10 = v4 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v3; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v3;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
      size_t v29 = 8;
      size_t v30 = 16;
      size_t v31 = std::min(v11, v30);
      size_t v32 = std::max(v31, v29);
      size_t v33 = v32 - v29;
      size_t v34 = __riscv_vsetvl_e32m1(v33);
      size_t v35 = v14 + v29;
      float* v36 = &v1[v35];
      vfloat32m1_t v37 = __riscv_vle32_v_f32m1(v36, v34);
      size_t v38 = v17 + v29;
      float* v39 = &v2[v38];
      __riscv_vse32_v_f32m1(v39, v37, v34);
    }
  }
  return;
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR10(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 10;
  for (size_t i9 = v6; i9 < v3; i9 += v8) {
    size_t v10 = v3 - i9;
    size_t v11 = std::min(v10, v8);
    for (size_t j12 = v6; j12 < v4; j12 += v7) {
      size_t v13 = j12 * v5;
      size_t v14 = v13 + i9;
      size_t v15 = i9 * v4;
      size_t v16 = j12 * v8;
      size_t v17 = v15 + v16;
      size_t v18 = 0;
      size_t v19 = 8;
      size_t v20 = std::min(v11, v19);
      size_t v21 = std::max(v20, v18);
      size_t v22 = v21 - v18;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      size_t v24 = v14 + v18;
      float* v25 = &v1[v24];
      vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v23);
      size_t v27 = v17 + v18;
      float* v28 = &v2[v27];
      __riscv_vse32_v_f32m1(v28, v26, v23);
      size_t v29 = 8;
      size_t v30 = 16;
      size_t v31 = std::min(v11, v30);
      size_t v32 = std::max(v31, v29);
      size_t v33 = v32 - v29;
      size_t v34 = __riscv_vsetvl_e32m1(v33);
      size_t v35 = v14 + v29;
      float* v36 = &v1[v35];
      vfloat32m1_t v37 = __riscv_vle32_v_f32m1(v36, v34);
      size_t v38 = v17 + v29;
      float* v39 = &v2[v38];
      __riscv_vse32_v_f32m1(v39, v37, v34);
    }
  }
  return;
}
void pack_RB_v_else_AXPY_FP32_Pack_fp32_MR10(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 10;
  for (size_t i10 = v6; i10 < v3; i10 += v9) {
    size_t v11 = v3 - i10;
    size_t v12 = std::min(v11, v9);
    size_t v13 = v5 * v8;
    for (size_t j14 = v6; j14 < v4; j14 += v7) {
      size_t v15 = i10 * v5;
      size_t v16 = v15 + j14;
      size_t v17 = i10 * v4;
      size_t v18 = j14 * v9;
      size_t v19 = v17 + v18;
      size_t v20 = 0;
      size_t v21 = 8;
      size_t v22 = std::min(v12, v21);
      size_t v23 = std::max(v22, v20);
      size_t v24 = v23 - v20;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      size_t v26 = v20 * v5;
      size_t v27 = v16 + v26;
      float* v28 = &v1[v27];
      vfloat32m1_t v29 = __riscv_vlse32_v_f32m1(v28, v13, v25);
      size_t v30 = v19 + v20;
      float* v31 = &v2[v30];
      __riscv_vse32_v_f32m1(v31, v29, v25);
      size_t v32 = 8;
      size_t v33 = 16;
      size_t v34 = std::min(v12, v33);
      size_t v35 = std::max(v34, v32);
      size_t v36 = v35 - v32;
      size_t v37 = __riscv_vsetvl_e32m1(v36);
      size_t v38 = v32 * v5;
      size_t v39 = v16 + v38;
      float* v40 = &v1[v39];
      vfloat32m1_t v41 = __riscv_vlse32_v_f32m1(v40, v13, v37);
      size_t v42 = v19 + v32;
      float* v43 = &v2[v42];
      __riscv_vse32_v_f32m1(v43, v41, v37);
    }
  }
  return;
}


}
