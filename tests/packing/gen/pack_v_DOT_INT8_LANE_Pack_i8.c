#include <riscv_vector.h>
#include <algorithm>

extern "C" {
void pack_CB_v_if_DOT_INT8_LANE_Pack_i8_RR16(int8_t* v1, int8_t* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 4;
  size_t v9 = 16;
  size_t v10 = 32;
  for (size_t i11 = v6; i11 < v4; i11 += v9) {
    size_t v12 = v4 - i11;
    size_t v13 = std::min(v12, v9);
    for (size_t j14 = v6; j14 < v3; j14 += v10) {
      size_t v15 = v3 - j14;
      size_t v16 = std::min(v15, v10);
      size_t v17 = __riscv_vsetvl_e8m1(v13);
      size_t v18 = v5 * v7;
      size_t v19 = v16 / v8;
      size_t v20 = v19 * v8;
      for (size_t k21 = v6; k21 < v20; k21 += v8) {
        size_t v22 = i11 * v5;
        size_t v23 = j14 + k21;
        size_t v24 = v22 + v23;
        size_t v25 = 1;
        size_t v26 = 2;
        size_t v27 = 3;
        size_t v28 = v24 + v25;
        size_t v29 = v24 + v26;
        size_t v30 = v24 + v27;
        int8_t* v31 = &v1[v24];
        vint8m1_t v32 = __riscv_vlse8_v_i8m1(v31, v18, v17);
        int8_t* v33 = &v1[v28];
        vint8m1_t v34 = __riscv_vlse8_v_i8m1(v33, v18, v17);
        int8_t* v35 = &v1[v29];
        vint8m1_t v36 = __riscv_vlse8_v_i8m1(v35, v18, v17);
        int8_t* v37 = &v1[v30];
        vint8m1_t v38 = __riscv_vlse8_v_i8m1(v37, v18, v17);
        size_t v39 = 31;
        size_t v40 = v3 + v39;
        size_t v41 = v40 / v10;
        size_t v42 = v41 * v10;
        size_t v43 = i11 / v9;
        size_t v44 = v42 * v9;
        size_t v45 = v43 * v44;
        size_t v46 = j14 * v9;
        size_t v47 = k21 * v9;
        size_t v48 = v45 + v46;
        size_t v49 = v48 + v47;
        size_t v50 = v49 + v25;
        size_t v51 = v49 + v26;
        size_t v52 = v49 + v27;
        int8_t* v53 = &v2[v49];
        __riscv_vsse8_v_i8m1(v53, v8, v32, v17);
        int8_t* v54 = &v2[v50];
        __riscv_vsse8_v_i8m1(v54, v8, v34, v17);
        int8_t* v55 = &v2[v51];
        __riscv_vsse8_v_i8m1(v55, v8, v36, v17);
        int8_t* v56 = &v2[v52];
        __riscv_vsse8_v_i8m1(v56, v8, v38, v17);
      }
    }
  }
  return;
}

void pack_CB_v_else_DOT_INT8_LANE_Pack_i8_RR16(int8_t* v1, int8_t* v2, size_t v3, size_t v4, size_t v5) {
  return;
}

void pack_RB_v_if_DOT_INT8_LANE_Pack_i8_RR16(int8_t* v1, int8_t* v2, size_t v3, size_t v4, size_t v5) {
  size_t v6 = 0;
  size_t v7 = 1;
  size_t v8 = 16;
  size_t v9 = 32;
  size_t v10 = 1;
  for (size_t i11 = v6; i11 < v3; i11 += v8) {
    size_t v12 = v3 - i11;
    size_t v13 = std::min(v12, v8);
    for (size_t j14 = v6; j14 < v4; j14 += v9) {
      size_t v15 = v4 - j14;
      size_t v16 = std::min(v15, v9);
      size_t v17 = __riscv_vsetvl_e8m1(v16);
      size_t v18 = v5 * v10;
      for (size_t k19 = v6; k19 < v13; k19 += v7) {
        size_t v20 = i11 + k19;
        size_t v21 = j14 * v5;
        size_t v22 = v21 + v20;
        int8_t* v23 = &v1[v22];
        vint8m1_t v24 = __riscv_vlse8_v_i8m1(v23, v18, v17);
        size_t v25 = 31;
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
        int8_t* v36 = &v2[v35];
        __riscv_vse8_v_i8m1(v36, v24, v17);
      }
    }
  }
  return;
}

void pack_RB_v_else_DOT_INT8_LANE_Pack_i8_RR16(int8_t* v1, int8_t* v2, size_t v3, size_t v4, size_t v5) {
  return;
}



}
