#include "gemm_headers.h"

extern "C" {

void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR1(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 1;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR1(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR1(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 1;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR1(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR2(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 2;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR2(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR2(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 2;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR2(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR3(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 3;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR3(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR3(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 3;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR3(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR4(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR5(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 5;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR5(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR5(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 5;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR5(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR6(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 6;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR6(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR6(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 6;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR6(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR7(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 7;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR7(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR7(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 7;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR7(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 8;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 8;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR8(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR9(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 9;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR9(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR9(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 9;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR9(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR10(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 10;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR10(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR10(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 10;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR10(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR11(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 11;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR11(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR11(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 11;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR11(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR12(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 12;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR12(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR12(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 12;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR12(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR13(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 13;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR13(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR13(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 13;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR13(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR14(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 14;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR14(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR14(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 14;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR14(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR15(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 15;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR15(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR15(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 15;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR15(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR16(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 16;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR16(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR16(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 16;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR16(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR17(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 17;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR17(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR17(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 17;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR17(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR18(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 18;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR18(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR18(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 18;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR18(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR19(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 19;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR19(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR19(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 19;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR19(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR20(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 20;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR20(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR20(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 20;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR20(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR21(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 21;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR21(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR21(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 21;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR21(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR22(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 22;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR22(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR22(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 22;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR22(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR23(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 23;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR23(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR23(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 23;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR23(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR24(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 24;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR24(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR24(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 24;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR24(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR25(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 25;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR25(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR25(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 25;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR25(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR26(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 26;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR26(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR26(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 26;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR26(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR27(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 27;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR27(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR27(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 27;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR27(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR28(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 28;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR28(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR28(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 28;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR28(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR29(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 29;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR29(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR29(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 29;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR29(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR30(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 30;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR30(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR30(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 30;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR30(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR31(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 31;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR31(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR31(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 31;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR31(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR32(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 32;
  size_t v9 = 4;
  for (size_t i10 = v6; i10 < v4; i10 += v8) {
    size_t v11 = v4 - i10;
    size_t v12 = std::min(v11, v8);
    for (size_t j13 = v6; j13 < v3; j13 += v9) {
      size_t v14 = v3 - j13;
      size_t v15 = std::min(v14, v9);
      size_t v16 = __riscv_vsetvl_e32m1(v15);
      for (size_t k17 = v6; k17 < v12; k17 += v7) {
        size_t v18 = i10 + k17;
        size_t v19 = v18 * v5;
        size_t v20 = v19 + j13;
        float* v21 = &v1[v20];
        vfloat32m1_t v22 = __riscv_vle32_v_f32m1(v21, v16);
        size_t v23 = 3;
        size_t v24 = v3 + v23;
        size_t v25 = v24 / v9;
        size_t v26 = v25 * v9;
        size_t v27 = i10 / v8;
        size_t v28 = v26 * v8;
        size_t v29 = v27 * v28;
        size_t v30 = j13 / v9;
        size_t v31 = v8 * v9;
        size_t v32 = v30 * v31;
        size_t v33 = k17 * v9;
        size_t v34 = v29 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v22, v16);
      }
    }
  }
  return;
}
void pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR32(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}
void pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR32(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 32;
  size_t v9 = 4;
  size_t v10 = 4;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e32m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        float* v23 = &v1[v22];
        vfloat32m1_t v24 = __riscv_vlse32_v_f32m1(v23, v18, v17);
        size_t v25 = 3;
        size_t v26 = v4 + v25;
        size_t v27 = v26 / v9;
        size_t v28 = v27 * v9;
        size_t v29 = i11 / v8;
        size_t v30 = v28 * v8;
        size_t v31 = v29 * v30;
        size_t v32 = j14 * v8;
        size_t v33 = k19 * v9;
        size_t v34 = v31 + v32;
        size_t v35 = v34 + v33;
        float* v36 = &v2[v35];
        __riscv_vse32_v_f32m1(v36, v24, v17);
      }
    }
  }
  return;
}
void pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR32(float* v1, float* v2, size_t v3, size_t v4, size_t v5) {
  return;
}


}
