void test_mixed_precision(int8_t* v1, int8_t* v2, int32_t* v3, size_t v4) {
  size_t v5 = 0;
  int16_t v6 = 0;
  int32_t v7 = 0;
  size_t v8 = __riscv_vsetvl_e8mf4(v4);
  int8_t* v9 = &v1[v5];
  vint8mf4_t v10 = __riscv_vle8_v_i8mf4(v9, v8);
  int8_t* v11 = &v2[v5];
  vint8mf4_t v12 = __riscv_vle8_v_i8mf4(v11, v8);
  vint16mf2_t v13 = __riscv_vmv_v_x_i16mf2(v6, v8);
  vint16mf2_t v14 = __riscv_vwmacc_vv_i16mf2(v13, v10, v12, v8);
  vint32m1_t v15 = __riscv_vmv_v_x_i32m1(v7, v8);
  vint32m1_t v16 = __riscv_vwadd_wv_i32m1(v15, v14, v8);
  int32_t* v17 = &v3[v5];
  __riscv_vse32_v_i32m1(v17, v16, v8);
  return;
}

