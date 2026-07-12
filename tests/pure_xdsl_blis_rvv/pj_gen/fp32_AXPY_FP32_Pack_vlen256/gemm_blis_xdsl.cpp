#include "gemm_headers.h"

extern "C" {

void gemm_blis_v_fp32_MR8_NR8(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR8(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR8(v33, v12, v30, v23, v6);
        int32_t v34 = 8;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 8;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 8;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 8;
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
            gemm_microkernel_v_dispatch_fp32_MR8_NR8(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR8_NR7(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR7(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR8(v33, v12, v30, v23, v6);
        int32_t v34 = 7;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 7;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 8;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 8;
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
            gemm_microkernel_v_dispatch_fp32_MR8_NR7(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR8_NR6(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR6(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR8(v33, v12, v30, v23, v6);
        int32_t v34 = 6;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 6;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 8;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 8;
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
            gemm_microkernel_v_dispatch_fp32_MR8_NR6(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR8_NR5(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR5(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR8(v33, v12, v30, v23, v6);
        int32_t v34 = 5;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 5;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 8;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 8;
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
            gemm_microkernel_v_dispatch_fp32_MR8_NR5(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR8_NR4(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR4(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR8(v33, v12, v30, v23, v6);
        int32_t v34 = 4;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 4;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 8;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 8;
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
            gemm_microkernel_v_dispatch_fp32_MR8_NR4(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR8_NR3(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR3(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR8(v33, v12, v30, v23, v6);
        int32_t v34 = 3;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 3;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 8;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 8;
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
            gemm_microkernel_v_dispatch_fp32_MR8_NR3(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR8_NR2(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR2(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR8(v33, v12, v30, v23, v6);
        int32_t v34 = 2;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 2;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 8;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 8;
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
            gemm_microkernel_v_dispatch_fp32_MR8_NR2(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR7_NR8(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR8(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR7(v33, v12, v30, v23, v6);
        int32_t v34 = 8;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 8;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 7;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 7;
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
            gemm_microkernel_v_dispatch_fp32_MR7_NR8(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR7_NR7(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR7(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR7(v33, v12, v30, v23, v6);
        int32_t v34 = 7;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 7;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 7;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 7;
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
            gemm_microkernel_v_dispatch_fp32_MR7_NR7(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR7_NR6(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR6(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR7(v33, v12, v30, v23, v6);
        int32_t v34 = 6;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 6;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 7;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 7;
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
            gemm_microkernel_v_dispatch_fp32_MR7_NR6(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR7_NR5(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR5(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR7(v33, v12, v30, v23, v6);
        int32_t v34 = 5;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 5;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 7;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 7;
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
            gemm_microkernel_v_dispatch_fp32_MR7_NR5(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR7_NR4(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR4(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR7(v33, v12, v30, v23, v6);
        int32_t v34 = 4;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 4;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 7;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 7;
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
            gemm_microkernel_v_dispatch_fp32_MR7_NR4(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR7_NR3(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR3(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR7(v33, v12, v30, v23, v6);
        int32_t v34 = 3;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 3;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 7;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 7;
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
            gemm_microkernel_v_dispatch_fp32_MR7_NR3(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR7_NR2(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR2(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR7(v33, v12, v30, v23, v6);
        int32_t v34 = 2;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 2;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 7;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 7;
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
            gemm_microkernel_v_dispatch_fp32_MR7_NR2(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR6_NR8(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR8(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR6(v33, v12, v30, v23, v6);
        int32_t v34 = 8;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 8;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 6;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 6;
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
            gemm_microkernel_v_dispatch_fp32_MR6_NR8(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR6_NR7(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR7(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR6(v33, v12, v30, v23, v6);
        int32_t v34 = 7;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 7;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 6;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 6;
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
            gemm_microkernel_v_dispatch_fp32_MR6_NR7(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR6_NR6(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR6(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR6(v33, v12, v30, v23, v6);
        int32_t v34 = 6;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 6;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 6;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 6;
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
            gemm_microkernel_v_dispatch_fp32_MR6_NR6(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR6_NR5(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR5(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR6(v33, v12, v30, v23, v6);
        int32_t v34 = 5;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 5;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 6;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 6;
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
            gemm_microkernel_v_dispatch_fp32_MR6_NR5(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR6_NR4(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR4(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR6(v33, v12, v30, v23, v6);
        int32_t v34 = 4;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 4;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 6;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 6;
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
            gemm_microkernel_v_dispatch_fp32_MR6_NR4(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR6_NR3(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR3(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR6(v33, v12, v30, v23, v6);
        int32_t v34 = 3;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 3;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 6;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 6;
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
            gemm_microkernel_v_dispatch_fp32_MR6_NR3(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR6_NR2(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR2(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR6(v33, v12, v30, v23, v6);
        int32_t v34 = 2;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 2;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 6;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 6;
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
            gemm_microkernel_v_dispatch_fp32_MR6_NR2(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR5_NR8(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR8(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR5(v33, v12, v30, v23, v6);
        int32_t v34 = 8;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 8;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 5;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 5;
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
            gemm_microkernel_v_dispatch_fp32_MR5_NR8(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR5_NR7(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR7(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR5(v33, v12, v30, v23, v6);
        int32_t v34 = 7;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 7;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 5;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 5;
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
            gemm_microkernel_v_dispatch_fp32_MR5_NR7(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR5_NR6(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR6(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR5(v33, v12, v30, v23, v6);
        int32_t v34 = 6;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 6;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 5;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 5;
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
            gemm_microkernel_v_dispatch_fp32_MR5_NR6(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR5_NR5(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR5(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR5(v33, v12, v30, v23, v6);
        int32_t v34 = 5;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 5;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 5;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 5;
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
            gemm_microkernel_v_dispatch_fp32_MR5_NR5(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR5_NR4(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR4(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR5(v33, v12, v30, v23, v6);
        int32_t v34 = 4;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 4;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 5;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 5;
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
            gemm_microkernel_v_dispatch_fp32_MR5_NR4(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR5_NR3(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR3(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR5(v33, v12, v30, v23, v6);
        int32_t v34 = 3;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 3;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 5;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 5;
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
            gemm_microkernel_v_dispatch_fp32_MR5_NR3(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR5_NR2(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR2(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR5(v33, v12, v30, v23, v6);
        int32_t v34 = 2;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 2;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 5;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 5;
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
            gemm_microkernel_v_dispatch_fp32_MR5_NR2(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR4_NR8(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR8(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR4(v33, v12, v30, v23, v6);
        int32_t v34 = 8;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 8;
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
            gemm_microkernel_v_dispatch_fp32_MR4_NR8(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR4_NR7(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR7(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR4(v33, v12, v30, v23, v6);
        int32_t v34 = 7;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 7;
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
            gemm_microkernel_v_dispatch_fp32_MR4_NR7(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR4_NR6(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR6(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR4(v33, v12, v30, v23, v6);
        int32_t v34 = 6;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 6;
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
            gemm_microkernel_v_dispatch_fp32_MR4_NR6(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR4_NR5(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR5(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR4(v33, v12, v30, v23, v6);
        int32_t v34 = 5;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 5;
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
            gemm_microkernel_v_dispatch_fp32_MR4_NR5(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR4_NR4(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR4(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR4(v33, v12, v30, v23, v6);
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
void gemm_blis_v_fp32_MR4_NR3(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR3(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR4(v33, v12, v30, v23, v6);
        int32_t v34 = 3;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 3;
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
            gemm_microkernel_v_dispatch_fp32_MR4_NR3(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR4_NR2(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR2(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR4(v33, v12, v30, v23, v6);
        int32_t v34 = 2;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 2;
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
            gemm_microkernel_v_dispatch_fp32_MR4_NR2(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR3_NR8(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR8(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR3(v33, v12, v30, v23, v6);
        int32_t v34 = 8;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 8;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 3;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 3;
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
            gemm_microkernel_v_dispatch_fp32_MR3_NR8(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR3_NR7(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR7(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR3(v33, v12, v30, v23, v6);
        int32_t v34 = 7;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 7;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 3;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 3;
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
            gemm_microkernel_v_dispatch_fp32_MR3_NR7(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR3_NR6(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR6(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR3(v33, v12, v30, v23, v6);
        int32_t v34 = 6;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 6;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 3;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 3;
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
            gemm_microkernel_v_dispatch_fp32_MR3_NR6(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR3_NR5(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR5(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR3(v33, v12, v30, v23, v6);
        int32_t v34 = 5;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 5;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 3;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 3;
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
            gemm_microkernel_v_dispatch_fp32_MR3_NR5(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR3_NR4(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR4(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR3(v33, v12, v30, v23, v6);
        int32_t v34 = 4;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 4;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 3;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 3;
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
            gemm_microkernel_v_dispatch_fp32_MR3_NR4(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR3_NR3(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR3(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR3(v33, v12, v30, v23, v6);
        int32_t v34 = 3;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 3;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 3;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 3;
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
            gemm_microkernel_v_dispatch_fp32_MR3_NR3(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR3_NR2(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR2(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR3(v33, v12, v30, v23, v6);
        int32_t v34 = 2;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 2;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 3;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 3;
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
            gemm_microkernel_v_dispatch_fp32_MR3_NR2(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR2_NR8(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR8(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR2(v33, v12, v30, v23, v6);
        int32_t v34 = 8;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 8;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 2;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 2;
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
            gemm_microkernel_v_dispatch_fp32_MR2_NR8(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR2_NR7(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR7(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR2(v33, v12, v30, v23, v6);
        int32_t v34 = 7;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 7;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 2;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 2;
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
            gemm_microkernel_v_dispatch_fp32_MR2_NR7(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR2_NR6(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR6(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR2(v33, v12, v30, v23, v6);
        int32_t v34 = 6;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 6;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 2;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 2;
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
            gemm_microkernel_v_dispatch_fp32_MR2_NR6(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR2_NR5(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR5(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR2(v33, v12, v30, v23, v6);
        int32_t v34 = 5;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 5;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 2;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 2;
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
            gemm_microkernel_v_dispatch_fp32_MR2_NR5(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR2_NR4(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR4(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR2(v33, v12, v30, v23, v6);
        int32_t v34 = 4;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 4;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 2;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 2;
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
            gemm_microkernel_v_dispatch_fp32_MR2_NR4(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR2_NR3(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR3(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR2(v33, v12, v30, v23, v6);
        int32_t v34 = 3;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 3;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 2;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 2;
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
            gemm_microkernel_v_dispatch_fp32_MR2_NR3(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}
void gemm_blis_v_fp32_MR2_NR2(int32_t v1, int32_t v2, int32_t v3, float v4, float* v5, int32_t v6, float* v7, int32_t v8, float v9, float* v10, int32_t v11, float* v12, float* v13, int32_t v14, int32_t v15, int32_t v16) {
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
      pack_CB_v_if_fp32_NR2(v26, v13, v23, v20, v8);
      float v27 = GET_BETA(j21, v9);
      for (int32_t k28 = v17; k28 < v1; k28 += v14) {
        int32_t v29 = v1 - k28;
        int32_t v30 = std::min(v29, v14);
        int32_t v31 = j21 * v6;
        int32_t v32 = v31 + k28;
        float* v33 = GET_PTR(v5, v32);
        pack_RB_v_if_fp32_MR2(v33, v12, v30, v23, v6);
        int32_t v34 = 2;
        for (int32_t l35 = v17; l35 < v20; l35 += v34) {
          int32_t v36 = v20 - l35;
          int32_t v37 = 2;
          int32_t v38 = std::min(v36, v37);
          int32_t v39 = 2;
          for (int32_t m40 = v17; m40 < v30; m40 += v39) {
            int32_t v41 = v30 - m40;
            int32_t v42 = 2;
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
            gemm_microkernel_v_dispatch_fp32_MR2_NR2(v43, v38, v23, v4, v50, v52, v27, v48, v11);
          }
        }
      }
    }
  }
  return;
}


}
