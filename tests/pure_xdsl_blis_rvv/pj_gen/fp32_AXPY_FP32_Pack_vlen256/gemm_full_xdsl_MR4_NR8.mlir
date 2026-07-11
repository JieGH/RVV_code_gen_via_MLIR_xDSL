"builtin.module"() ({
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x1_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%20, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%21 : index):
      %22 = "emitc.mul"(%21, %3) : (index, index) -> index
      %23 = "emitc.constant"() {value = 1 : index} : () -> index
      %24 = "emitc.subscript"(%2, %22) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %25 = "emitc.apply"(%24) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %26 = "emitc.call_opaque"(%25, %23) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %27 = "emitc.mul"(%21, %5) : (index, index) -> index
      %28 = "emitc.load"(%20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.subscript"(%4, %27) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %30 = "emitc.load"(%29) : (!emitc.lvalue<f32>) -> f32
      %31 = "emitc.call_opaque"(%28, %30, %26, %23) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%20, %31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %32 = "emitc.load"(%20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %34 = "emitc.apply"(%33) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%34, %32, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x1_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%23 : index):
      %24 = "emitc.mul"(%23, %3) : (index, index) -> index
      %25 = "emitc.constant"() {value = 1 : index} : () -> index
      %26 = "emitc.subscript"(%2, %24) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %28 = "emitc.call_opaque"(%27, %25) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.mul"(%23, %5) : (index, index) -> index
      %30 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %31 = "emitc.subscript"(%4, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %32 = "emitc.load"(%31) : (!emitc.lvalue<f32>) -> f32
      %33 = "emitc.call_opaque"(%30, %32, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%22, %33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %34 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %35 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %36 = "emitc.apply"(%35) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%36, %34, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x2_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%21, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%23 : index):
      %24 = "emitc.mul"(%23, %3) : (index, index) -> index
      %25 = "emitc.constant"() {value = 1 : index} : () -> index
      %26 = "emitc.subscript"(%2, %24) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %28 = "emitc.call_opaque"(%27, %25) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.mul"(%23, %5) : (index, index) -> index
      %30 = "emitc.add"(%29, %11) : (index, index) -> index
      %31 = "emitc.load"(%21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %32 = "emitc.subscript"(%4, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %33 = "emitc.load"(%32) : (!emitc.lvalue<f32>) -> f32
      %34 = "emitc.call_opaque"(%31, %33, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %36 = "emitc.subscript"(%4, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.load"(%36) : (!emitc.lvalue<f32>) -> f32
      %38 = "emitc.call_opaque"(%35, %37, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%21, %34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%22, %38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %39 = "emitc.load"(%21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %40 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %42 = "emitc.apply"(%41) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%42, %39, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %43 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %44 = "emitc.apply"(%43) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%44, %40, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x2_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%27 : index):
      %28 = "emitc.mul"(%27, %3) : (index, index) -> index
      %29 = "emitc.constant"() {value = 1 : index} : () -> index
      %30 = "emitc.subscript"(%2, %28) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %32 = "emitc.call_opaque"(%31, %29) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %33 = "emitc.mul"(%27, %5) : (index, index) -> index
      %34 = "emitc.add"(%33, %11) : (index, index) -> index
      %35 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %36 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.load"(%36) : (!emitc.lvalue<f32>) -> f32
      %38 = "emitc.call_opaque"(%35, %37, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %40 = "emitc.subscript"(%4, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %41 = "emitc.load"(%40) : (!emitc.lvalue<f32>) -> f32
      %42 = "emitc.call_opaque"(%39, %41, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%25, %38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %43 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %44 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %46 = "emitc.apply"(%45) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%46, %43, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %47 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %48 = "emitc.apply"(%47) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%48, %44, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x3_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %23 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%23, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%25 : index):
      %26 = "emitc.mul"(%25, %3) : (index, index) -> index
      %27 = "emitc.constant"() {value = 1 : index} : () -> index
      %28 = "emitc.subscript"(%2, %26) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %29 = "emitc.apply"(%28) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %30 = "emitc.call_opaque"(%29, %27) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %31 = "emitc.mul"(%25, %5) : (index, index) -> index
      %32 = "emitc.add"(%31, %11) : (index, index) -> index
      %33 = "emitc.add"(%31, %12) : (index, index) -> index
      %34 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.subscript"(%4, %31) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %36 = "emitc.load"(%35) : (!emitc.lvalue<f32>) -> f32
      %37 = "emitc.call_opaque"(%34, %36, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.subscript"(%4, %32) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %40 = "emitc.load"(%39) : (!emitc.lvalue<f32>) -> f32
      %41 = "emitc.call_opaque"(%38, %40, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %43 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %44 = "emitc.load"(%43) : (!emitc.lvalue<f32>) -> f32
      %45 = "emitc.call_opaque"(%42, %44, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%22, %37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%23, %41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%24, %45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %46 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %47 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %48 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %49 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %50 = "emitc.apply"(%49) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%50, %46, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %51 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %52 = "emitc.apply"(%51) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%52, %47, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %53 = "emitc.mul"(%12, %8) : (index, index) -> index
    %54 = "emitc.subscript"(%7, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %55 = "emitc.apply"(%54) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%55, %48, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x3_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%32 : index):
      %33 = "emitc.mul"(%32, %3) : (index, index) -> index
      %34 = "emitc.constant"() {value = 1 : index} : () -> index
      %35 = "emitc.subscript"(%2, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %36 = "emitc.apply"(%35) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %37 = "emitc.call_opaque"(%36, %34) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.mul"(%32, %5) : (index, index) -> index
      %39 = "emitc.add"(%38, %11) : (index, index) -> index
      %40 = "emitc.add"(%38, %12) : (index, index) -> index
      %41 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %43 = "emitc.load"(%42) : (!emitc.lvalue<f32>) -> f32
      %44 = "emitc.call_opaque"(%41, %43, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %46 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %47 = "emitc.load"(%46) : (!emitc.lvalue<f32>) -> f32
      %48 = "emitc.call_opaque"(%45, %47, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%29, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %53 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %54 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %55 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %56 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %57 = "emitc.apply"(%56) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%57, %53, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %58 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %59 = "emitc.apply"(%58) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%59, %54, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %60 = "emitc.mul"(%12, %8) : (index, index) -> index
    %61 = "emitc.subscript"(%7, %60) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %62 = "emitc.apply"(%61) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%62, %55, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x4_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%23, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%27 : index):
      %28 = "emitc.mul"(%27, %3) : (index, index) -> index
      %29 = "emitc.constant"() {value = 1 : index} : () -> index
      %30 = "emitc.subscript"(%2, %28) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %32 = "emitc.call_opaque"(%31, %29) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %33 = "emitc.mul"(%27, %5) : (index, index) -> index
      %34 = "emitc.add"(%33, %11) : (index, index) -> index
      %35 = "emitc.add"(%33, %12) : (index, index) -> index
      %36 = "emitc.add"(%33, %13) : (index, index) -> index
      %37 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %39 = "emitc.load"(%38) : (!emitc.lvalue<f32>) -> f32
      %40 = "emitc.call_opaque"(%37, %39, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.subscript"(%4, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %43 = "emitc.load"(%42) : (!emitc.lvalue<f32>) -> f32
      %44 = "emitc.call_opaque"(%41, %43, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %46 = "emitc.subscript"(%4, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %47 = "emitc.load"(%46) : (!emitc.lvalue<f32>) -> f32
      %48 = "emitc.call_opaque"(%45, %47, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%23, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%24, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%25, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %53 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %54 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %55 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %56 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %57 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %58 = "emitc.apply"(%57) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%58, %53, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %59 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %60 = "emitc.apply"(%59) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%60, %54, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %61 = "emitc.mul"(%12, %8) : (index, index) -> index
    %62 = "emitc.subscript"(%7, %61) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %63 = "emitc.apply"(%62) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%63, %55, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %64 = "emitc.mul"(%13, %8) : (index, index) -> index
    %65 = "emitc.subscript"(%7, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %66 = "emitc.apply"(%65) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%66, %56, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x4_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%33, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %34 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%34, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %35 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%35, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %36 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%36, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%37 : index):
      %38 = "emitc.mul"(%37, %3) : (index, index) -> index
      %39 = "emitc.constant"() {value = 1 : index} : () -> index
      %40 = "emitc.subscript"(%2, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %41 = "emitc.apply"(%40) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %42 = "emitc.call_opaque"(%41, %39) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %43 = "emitc.mul"(%37, %5) : (index, index) -> index
      %44 = "emitc.add"(%43, %11) : (index, index) -> index
      %45 = "emitc.add"(%43, %12) : (index, index) -> index
      %46 = "emitc.add"(%43, %13) : (index, index) -> index
      %47 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %49 = "emitc.load"(%48) : (!emitc.lvalue<f32>) -> f32
      %50 = "emitc.call_opaque"(%47, %49, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %53 = "emitc.load"(%52) : (!emitc.lvalue<f32>) -> f32
      %54 = "emitc.call_opaque"(%51, %53, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.load"(%35) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %57 = "emitc.load"(%56) : (!emitc.lvalue<f32>) -> f32
      %58 = "emitc.call_opaque"(%55, %57, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.load"(%36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %46) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%33, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%34, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%35, %58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%36, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %63 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %64 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.load"(%35) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %66 = "emitc.load"(%36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %67 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %68 = "emitc.apply"(%67) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%68, %63, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %69 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %70 = "emitc.apply"(%69) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%70, %64, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %71 = "emitc.mul"(%12, %8) : (index, index) -> index
    %72 = "emitc.subscript"(%7, %71) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %73 = "emitc.apply"(%72) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%73, %65, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %74 = "emitc.mul"(%13, %8) : (index, index) -> index
    %75 = "emitc.subscript"(%7, %74) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %76 = "emitc.apply"(%75) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%76, %66, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x5_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%29 : index):
      %30 = "emitc.mul"(%29, %3) : (index, index) -> index
      %31 = "emitc.constant"() {value = 1 : index} : () -> index
      %32 = "emitc.subscript"(%2, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %33 = "emitc.apply"(%32) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %34 = "emitc.call_opaque"(%33, %31) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.mul"(%29, %5) : (index, index) -> index
      %36 = "emitc.add"(%35, %11) : (index, index) -> index
      %37 = "emitc.add"(%35, %12) : (index, index) -> index
      %38 = "emitc.add"(%35, %13) : (index, index) -> index
      %39 = "emitc.add"(%35, %14) : (index, index) -> index
      %40 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.subscript"(%4, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %42 = "emitc.load"(%41) : (!emitc.lvalue<f32>) -> f32
      %43 = "emitc.call_opaque"(%40, %42, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %44 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.subscript"(%4, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %46 = "emitc.load"(%45) : (!emitc.lvalue<f32>) -> f32
      %47 = "emitc.call_opaque"(%44, %46, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.subscript"(%4, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %50 = "emitc.load"(%49) : (!emitc.lvalue<f32>) -> f32
      %51 = "emitc.call_opaque"(%48, %50, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %54 = "emitc.load"(%53) : (!emitc.lvalue<f32>) -> f32
      %55 = "emitc.call_opaque"(%52, %54, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %58 = "emitc.load"(%57) : (!emitc.lvalue<f32>) -> f32
      %59 = "emitc.call_opaque"(%56, %58, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%24, %43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%25, %47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %59) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %60 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %61 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %62 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %63 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %64 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %66 = "emitc.apply"(%65) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%66, %60, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %67 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %68 = "emitc.apply"(%67) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%68, %61, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %69 = "emitc.mul"(%12, %8) : (index, index) -> index
    %70 = "emitc.subscript"(%7, %69) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %71 = "emitc.apply"(%70) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%71, %62, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %72 = "emitc.mul"(%13, %8) : (index, index) -> index
    %73 = "emitc.subscript"(%7, %72) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %74 = "emitc.apply"(%73) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%74, %63, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %75 = "emitc.mul"(%14, %8) : (index, index) -> index
    %76 = "emitc.subscript"(%7, %75) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %77 = "emitc.apply"(%76) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%77, %64, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x5_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%37, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %38 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%38, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %39 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%39, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %40 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%40, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %41 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%41, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%42 : index):
      %43 = "emitc.mul"(%42, %3) : (index, index) -> index
      %44 = "emitc.constant"() {value = 1 : index} : () -> index
      %45 = "emitc.subscript"(%2, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %46 = "emitc.apply"(%45) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %47 = "emitc.call_opaque"(%46, %44) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.mul"(%42, %5) : (index, index) -> index
      %49 = "emitc.add"(%48, %11) : (index, index) -> index
      %50 = "emitc.add"(%48, %12) : (index, index) -> index
      %51 = "emitc.add"(%48, %13) : (index, index) -> index
      %52 = "emitc.add"(%48, %14) : (index, index) -> index
      %53 = "emitc.load"(%37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.subscript"(%4, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %55 = "emitc.load"(%54) : (!emitc.lvalue<f32>) -> f32
      %56 = "emitc.call_opaque"(%53, %55, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.load"(%38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.subscript"(%4, %49) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %59 = "emitc.load"(%58) : (!emitc.lvalue<f32>) -> f32
      %60 = "emitc.call_opaque"(%57, %59, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %61 = "emitc.load"(%39) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.subscript"(%4, %50) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %63 = "emitc.load"(%62) : (!emitc.lvalue<f32>) -> f32
      %64 = "emitc.call_opaque"(%61, %63, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %65 = "emitc.load"(%40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %51) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %52) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%37, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%38, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%39, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%40, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%41, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %73 = "emitc.load"(%37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %74 = "emitc.load"(%38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %75 = "emitc.load"(%39) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %76 = "emitc.load"(%40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %77 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %78 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %79 = "emitc.apply"(%78) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%79, %73, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %80 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %81 = "emitc.apply"(%80) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%81, %74, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %82 = "emitc.mul"(%12, %8) : (index, index) -> index
    %83 = "emitc.subscript"(%7, %82) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %84 = "emitc.apply"(%83) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%84, %75, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %85 = "emitc.mul"(%13, %8) : (index, index) -> index
    %86 = "emitc.subscript"(%7, %85) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %87 = "emitc.apply"(%86) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%87, %76, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %88 = "emitc.mul"(%14, %8) : (index, index) -> index
    %89 = "emitc.subscript"(%7, %88) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %77, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x6_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%31 : index):
      %32 = "emitc.mul"(%31, %3) : (index, index) -> index
      %33 = "emitc.constant"() {value = 1 : index} : () -> index
      %34 = "emitc.subscript"(%2, %32) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %36 = "emitc.call_opaque"(%35, %33) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %37 = "emitc.mul"(%31, %5) : (index, index) -> index
      %38 = "emitc.add"(%37, %11) : (index, index) -> index
      %39 = "emitc.add"(%37, %12) : (index, index) -> index
      %40 = "emitc.add"(%37, %13) : (index, index) -> index
      %41 = "emitc.add"(%37, %14) : (index, index) -> index
      %42 = "emitc.add"(%37, %15) : (index, index) -> index
      %43 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %44 = "emitc.subscript"(%4, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %45 = "emitc.load"(%44) : (!emitc.lvalue<f32>) -> f32
      %46 = "emitc.call_opaque"(%43, %45, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %47 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %49 = "emitc.load"(%48) : (!emitc.lvalue<f32>) -> f32
      %50 = "emitc.call_opaque"(%47, %49, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %53 = "emitc.load"(%52) : (!emitc.lvalue<f32>) -> f32
      %54 = "emitc.call_opaque"(%51, %53, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %57 = "emitc.load"(%56) : (!emitc.lvalue<f32>) -> f32
      %58 = "emitc.call_opaque"(%55, %57, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %64 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %65 = "emitc.load"(%64) : (!emitc.lvalue<f32>) -> f32
      %66 = "emitc.call_opaque"(%63, %65, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%25, %46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %67 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %68 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %69 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %70 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %71 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %72 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %73 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %74 = "emitc.apply"(%73) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%74, %67, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %75 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %76 = "emitc.apply"(%75) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%76, %68, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %77 = "emitc.mul"(%12, %8) : (index, index) -> index
    %78 = "emitc.subscript"(%7, %77) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %79 = "emitc.apply"(%78) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%79, %69, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %80 = "emitc.mul"(%13, %8) : (index, index) -> index
    %81 = "emitc.subscript"(%7, %80) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %82 = "emitc.apply"(%81) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%82, %70, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %83 = "emitc.mul"(%14, %8) : (index, index) -> index
    %84 = "emitc.subscript"(%7, %83) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %85 = "emitc.apply"(%84) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%85, %71, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %86 = "emitc.mul"(%15, %8) : (index, index) -> index
    %87 = "emitc.subscript"(%7, %86) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %88 = "emitc.apply"(%87) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%88, %72, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x6_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%41, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %42 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%42, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %43 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%43, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %44 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%44, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %45 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%45, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %46 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%46, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%47 : index):
      %48 = "emitc.mul"(%47, %3) : (index, index) -> index
      %49 = "emitc.constant"() {value = 1 : index} : () -> index
      %50 = "emitc.subscript"(%2, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.apply"(%50) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %52 = "emitc.call_opaque"(%51, %49) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.mul"(%47, %5) : (index, index) -> index
      %54 = "emitc.add"(%53, %11) : (index, index) -> index
      %55 = "emitc.add"(%53, %12) : (index, index) -> index
      %56 = "emitc.add"(%53, %13) : (index, index) -> index
      %57 = "emitc.add"(%53, %14) : (index, index) -> index
      %58 = "emitc.add"(%53, %15) : (index, index) -> index
      %59 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.load"(%42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %64 = "emitc.subscript"(%4, %54) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %65 = "emitc.load"(%64) : (!emitc.lvalue<f32>) -> f32
      %66 = "emitc.call_opaque"(%63, %65, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %67 = "emitc.load"(%43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %68 = "emitc.subscript"(%4, %55) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %69 = "emitc.load"(%68) : (!emitc.lvalue<f32>) -> f32
      %70 = "emitc.call_opaque"(%67, %69, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %71 = "emitc.load"(%44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %72 = "emitc.subscript"(%4, %56) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %73 = "emitc.load"(%72) : (!emitc.lvalue<f32>) -> f32
      %74 = "emitc.call_opaque"(%71, %73, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %75 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %76 = "emitc.subscript"(%4, %57) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %77 = "emitc.load"(%76) : (!emitc.lvalue<f32>) -> f32
      %78 = "emitc.call_opaque"(%75, %77, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %79 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %80 = "emitc.subscript"(%4, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %81 = "emitc.load"(%80) : (!emitc.lvalue<f32>) -> f32
      %82 = "emitc.call_opaque"(%79, %81, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%41, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%42, %66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%43, %70) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%44, %74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%45, %78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%46, %82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %83 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %84 = "emitc.load"(%42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %85 = "emitc.load"(%43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %86 = "emitc.load"(%44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %87 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %88 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %89 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %83, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %92 = "emitc.apply"(%91) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%92, %84, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %93 = "emitc.mul"(%12, %8) : (index, index) -> index
    %94 = "emitc.subscript"(%7, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%95, %85, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %96 = "emitc.mul"(%13, %8) : (index, index) -> index
    %97 = "emitc.subscript"(%7, %96) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %98 = "emitc.apply"(%97) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%98, %86, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %99 = "emitc.mul"(%14, %8) : (index, index) -> index
    %100 = "emitc.subscript"(%7, %99) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %87, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.mul"(%15, %8) : (index, index) -> index
    %103 = "emitc.subscript"(%7, %102) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %104 = "emitc.apply"(%103) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%104, %88, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x7_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %32 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%32, %25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%33 : index):
      %34 = "emitc.mul"(%33, %3) : (index, index) -> index
      %35 = "emitc.constant"() {value = 1 : index} : () -> index
      %36 = "emitc.subscript"(%2, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %38 = "emitc.call_opaque"(%37, %35) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.mul"(%33, %5) : (index, index) -> index
      %40 = "emitc.add"(%39, %11) : (index, index) -> index
      %41 = "emitc.add"(%39, %12) : (index, index) -> index
      %42 = "emitc.add"(%39, %13) : (index, index) -> index
      %43 = "emitc.add"(%39, %14) : (index, index) -> index
      %44 = "emitc.add"(%39, %15) : (index, index) -> index
      %45 = "emitc.add"(%39, %16) : (index, index) -> index
      %46 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %47 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %48 = "emitc.load"(%47) : (!emitc.lvalue<f32>) -> f32
      %49 = "emitc.call_opaque"(%46, %48, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %52 = "emitc.load"(%51) : (!emitc.lvalue<f32>) -> f32
      %53 = "emitc.call_opaque"(%50, %52, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %56 = "emitc.load"(%55) : (!emitc.lvalue<f32>) -> f32
      %57 = "emitc.call_opaque"(%54, %56, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %60 = "emitc.load"(%59) : (!emitc.lvalue<f32>) -> f32
      %61 = "emitc.call_opaque"(%58, %60, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %64 = "emitc.load"(%63) : (!emitc.lvalue<f32>) -> f32
      %65 = "emitc.call_opaque"(%62, %64, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %67 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %68 = "emitc.load"(%67) : (!emitc.lvalue<f32>) -> f32
      %69 = "emitc.call_opaque"(%66, %68, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %71 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %72 = "emitc.load"(%71) : (!emitc.lvalue<f32>) -> f32
      %73 = "emitc.call_opaque"(%70, %72, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%26, %49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %57) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %61) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %65) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %69) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%32, %73) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %74 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %75 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %76 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %77 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %78 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %79 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %80 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %81 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %82 = "emitc.apply"(%81) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%82, %74, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %83 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %84 = "emitc.apply"(%83) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%84, %75, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %85 = "emitc.mul"(%12, %8) : (index, index) -> index
    %86 = "emitc.subscript"(%7, %85) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %87 = "emitc.apply"(%86) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%87, %76, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %88 = "emitc.mul"(%13, %8) : (index, index) -> index
    %89 = "emitc.subscript"(%7, %88) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %77, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.mul"(%14, %8) : (index, index) -> index
    %92 = "emitc.subscript"(%7, %91) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %93 = "emitc.apply"(%92) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%93, %78, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %94 = "emitc.mul"(%15, %8) : (index, index) -> index
    %95 = "emitc.subscript"(%7, %94) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %96 = "emitc.apply"(%95) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%96, %79, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %97 = "emitc.mul"(%16, %8) : (index, index) -> index
    %98 = "emitc.subscript"(%7, %97) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %99 = "emitc.apply"(%98) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%99, %80, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x7_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.mul"(%16, %8) : (index, index) -> index
    %42 = "emitc.subscript"(%7, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %43 = "emitc.apply"(%42) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %44 = "emitc.call_opaque"(%43, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%45, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %46 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%46, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %47 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%47, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %48 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%48, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %49 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%49, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %50 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%50, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %51 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%51, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%52 : index):
      %53 = "emitc.mul"(%52, %3) : (index, index) -> index
      %54 = "emitc.constant"() {value = 1 : index} : () -> index
      %55 = "emitc.subscript"(%2, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %56 = "emitc.apply"(%55) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %57 = "emitc.call_opaque"(%56, %54) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.mul"(%52, %5) : (index, index) -> index
      %59 = "emitc.add"(%58, %11) : (index, index) -> index
      %60 = "emitc.add"(%58, %12) : (index, index) -> index
      %61 = "emitc.add"(%58, %13) : (index, index) -> index
      %62 = "emitc.add"(%58, %14) : (index, index) -> index
      %63 = "emitc.add"(%58, %15) : (index, index) -> index
      %64 = "emitc.add"(%58, %16) : (index, index) -> index
      %65 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %59) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %73 = "emitc.load"(%47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %74 = "emitc.subscript"(%4, %60) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %75 = "emitc.load"(%74) : (!emitc.lvalue<f32>) -> f32
      %76 = "emitc.call_opaque"(%73, %75, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %77 = "emitc.load"(%48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %78 = "emitc.subscript"(%4, %61) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %79 = "emitc.load"(%78) : (!emitc.lvalue<f32>) -> f32
      %80 = "emitc.call_opaque"(%77, %79, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %81 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %82 = "emitc.subscript"(%4, %62) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %83 = "emitc.load"(%82) : (!emitc.lvalue<f32>) -> f32
      %84 = "emitc.call_opaque"(%81, %83, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %85 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %86 = "emitc.subscript"(%4, %63) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %87 = "emitc.load"(%86) : (!emitc.lvalue<f32>) -> f32
      %88 = "emitc.call_opaque"(%85, %87, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %89 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %90 = "emitc.subscript"(%4, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %91 = "emitc.load"(%90) : (!emitc.lvalue<f32>) -> f32
      %92 = "emitc.call_opaque"(%89, %91, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%45, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%46, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%47, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%48, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%49, %84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%50, %88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%51, %92) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %93 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %94 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %95 = "emitc.load"(%47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %96 = "emitc.load"(%48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %97 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %98 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %99 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %100 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %93, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %103 = "emitc.apply"(%102) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%103, %94, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %104 = "emitc.mul"(%12, %8) : (index, index) -> index
    %105 = "emitc.subscript"(%7, %104) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %106 = "emitc.apply"(%105) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%106, %95, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %107 = "emitc.mul"(%13, %8) : (index, index) -> index
    %108 = "emitc.subscript"(%7, %107) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %109 = "emitc.apply"(%108) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%109, %96, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %110 = "emitc.mul"(%14, %8) : (index, index) -> index
    %111 = "emitc.subscript"(%7, %110) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %112 = "emitc.apply"(%111) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%112, %97, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %113 = "emitc.mul"(%15, %8) : (index, index) -> index
    %114 = "emitc.subscript"(%7, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %115 = "emitc.apply"(%114) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%115, %98, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %116 = "emitc.mul"(%16, %8) : (index, index) -> index
    %117 = "emitc.subscript"(%7, %116) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %118 = "emitc.apply"(%117) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%118, %99, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x8_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %26 = "emitc.call_opaque"(%9, %11) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %32 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%32, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %33 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%33, %25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %34 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%34, %26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%35 : index):
      %36 = "emitc.mul"(%35, %3) : (index, index) -> index
      %37 = "emitc.constant"() {value = 1 : index} : () -> index
      %38 = "emitc.subscript"(%2, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %40 = "emitc.call_opaque"(%39, %37) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.mul"(%35, %5) : (index, index) -> index
      %42 = "emitc.add"(%41, %11) : (index, index) -> index
      %43 = "emitc.add"(%41, %12) : (index, index) -> index
      %44 = "emitc.add"(%41, %13) : (index, index) -> index
      %45 = "emitc.add"(%41, %14) : (index, index) -> index
      %46 = "emitc.add"(%41, %15) : (index, index) -> index
      %47 = "emitc.add"(%41, %16) : (index, index) -> index
      %48 = "emitc.add"(%41, %17) : (index, index) -> index
      %49 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %55 = "emitc.load"(%54) : (!emitc.lvalue<f32>) -> f32
      %56 = "emitc.call_opaque"(%53, %55, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %59 = "emitc.load"(%58) : (!emitc.lvalue<f32>) -> f32
      %60 = "emitc.call_opaque"(%57, %59, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %61 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %63 = "emitc.load"(%62) : (!emitc.lvalue<f32>) -> f32
      %64 = "emitc.call_opaque"(%61, %63, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %65 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %46) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %73 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %74 = "emitc.subscript"(%4, %47) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %75 = "emitc.load"(%74) : (!emitc.lvalue<f32>) -> f32
      %76 = "emitc.call_opaque"(%73, %75, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %77 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %78 = "emitc.subscript"(%4, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %79 = "emitc.load"(%78) : (!emitc.lvalue<f32>) -> f32
      %80 = "emitc.call_opaque"(%77, %79, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%27, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%32, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%33, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%34, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %81 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %82 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %83 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %84 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %85 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %86 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %87 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %88 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %89 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %81, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %92 = "emitc.apply"(%91) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%92, %82, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %93 = "emitc.mul"(%12, %8) : (index, index) -> index
    %94 = "emitc.subscript"(%7, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%95, %83, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %96 = "emitc.mul"(%13, %8) : (index, index) -> index
    %97 = "emitc.subscript"(%7, %96) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %98 = "emitc.apply"(%97) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%98, %84, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %99 = "emitc.mul"(%14, %8) : (index, index) -> index
    %100 = "emitc.subscript"(%7, %99) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %85, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.mul"(%15, %8) : (index, index) -> index
    %103 = "emitc.subscript"(%7, %102) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %104 = "emitc.apply"(%103) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%104, %86, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %105 = "emitc.mul"(%16, %8) : (index, index) -> index
    %106 = "emitc.subscript"(%7, %105) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %107 = "emitc.apply"(%106) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%107, %87, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %108 = "emitc.mul"(%17, %8) : (index, index) -> index
    %109 = "emitc.subscript"(%7, %108) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %110 = "emitc.apply"(%109) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%110, %88, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_1x8_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.mul"(%16, %8) : (index, index) -> index
    %42 = "emitc.subscript"(%7, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %43 = "emitc.apply"(%42) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %44 = "emitc.call_opaque"(%43, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.mul"(%17, %8) : (index, index) -> index
    %46 = "emitc.subscript"(%7, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %47 = "emitc.apply"(%46) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %48 = "emitc.call_opaque"(%47, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %49 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%49, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %50 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%50, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %51 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%51, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %52 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%52, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %53 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%53, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %54 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%54, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %55 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%55, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %56 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%56, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%57 : index):
      %58 = "emitc.mul"(%57, %3) : (index, index) -> index
      %59 = "emitc.constant"() {value = 1 : index} : () -> index
      %60 = "emitc.subscript"(%2, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.apply"(%60) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %62 = "emitc.call_opaque"(%61, %59) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.mul"(%57, %5) : (index, index) -> index
      %64 = "emitc.add"(%63, %11) : (index, index) -> index
      %65 = "emitc.add"(%63, %12) : (index, index) -> index
      %66 = "emitc.add"(%63, %13) : (index, index) -> index
      %67 = "emitc.add"(%63, %14) : (index, index) -> index
      %68 = "emitc.add"(%63, %15) : (index, index) -> index
      %69 = "emitc.add"(%63, %16) : (index, index) -> index
      %70 = "emitc.add"(%63, %17) : (index, index) -> index
      %71 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %72 = "emitc.subscript"(%4, %63) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %73 = "emitc.load"(%72) : (!emitc.lvalue<f32>) -> f32
      %74 = "emitc.call_opaque"(%71, %73, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %75 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %76 = "emitc.subscript"(%4, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %77 = "emitc.load"(%76) : (!emitc.lvalue<f32>) -> f32
      %78 = "emitc.call_opaque"(%75, %77, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %79 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %80 = "emitc.subscript"(%4, %65) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %81 = "emitc.load"(%80) : (!emitc.lvalue<f32>) -> f32
      %82 = "emitc.call_opaque"(%79, %81, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %83 = "emitc.load"(%52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %84 = "emitc.subscript"(%4, %66) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %85 = "emitc.load"(%84) : (!emitc.lvalue<f32>) -> f32
      %86 = "emitc.call_opaque"(%83, %85, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %87 = "emitc.load"(%53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %88 = "emitc.subscript"(%4, %67) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %89 = "emitc.load"(%88) : (!emitc.lvalue<f32>) -> f32
      %90 = "emitc.call_opaque"(%87, %89, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %91 = "emitc.load"(%54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %92 = "emitc.subscript"(%4, %68) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %93 = "emitc.load"(%92) : (!emitc.lvalue<f32>) -> f32
      %94 = "emitc.call_opaque"(%91, %93, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %95 = "emitc.load"(%55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %96 = "emitc.subscript"(%4, %69) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %97 = "emitc.load"(%96) : (!emitc.lvalue<f32>) -> f32
      %98 = "emitc.call_opaque"(%95, %97, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %99 = "emitc.load"(%56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %100 = "emitc.subscript"(%4, %70) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %101 = "emitc.load"(%100) : (!emitc.lvalue<f32>) -> f32
      %102 = "emitc.call_opaque"(%99, %101, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%49, %74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%50, %78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%51, %82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%52, %86) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%53, %90) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%54, %94) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%55, %98) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%56, %102) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %103 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %104 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %105 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %106 = "emitc.load"(%52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %107 = "emitc.load"(%53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %108 = "emitc.load"(%54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %109 = "emitc.load"(%55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %110 = "emitc.load"(%56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %111 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %112 = "emitc.apply"(%111) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%112, %103, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %113 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %114 = "emitc.apply"(%113) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%114, %104, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %115 = "emitc.mul"(%12, %8) : (index, index) -> index
    %116 = "emitc.subscript"(%7, %115) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %117 = "emitc.apply"(%116) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%117, %105, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %118 = "emitc.mul"(%13, %8) : (index, index) -> index
    %119 = "emitc.subscript"(%7, %118) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %120 = "emitc.apply"(%119) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%120, %106, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %121 = "emitc.mul"(%14, %8) : (index, index) -> index
    %122 = "emitc.subscript"(%7, %121) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %123 = "emitc.apply"(%122) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%123, %107, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %124 = "emitc.mul"(%15, %8) : (index, index) -> index
    %125 = "emitc.subscript"(%7, %124) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %126 = "emitc.apply"(%125) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%126, %108, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %127 = "emitc.mul"(%16, %8) : (index, index) -> index
    %128 = "emitc.subscript"(%7, %127) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %129 = "emitc.apply"(%128) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%129, %109, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %130 = "emitc.mul"(%17, %8) : (index, index) -> index
    %131 = "emitc.subscript"(%7, %130) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %132 = "emitc.apply"(%131) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%132, %110, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x1_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%20, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%21 : index):
      %22 = "emitc.mul"(%21, %3) : (index, index) -> index
      %23 = "emitc.constant"() {value = 2 : index} : () -> index
      %24 = "emitc.subscript"(%2, %22) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %25 = "emitc.apply"(%24) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %26 = "emitc.call_opaque"(%25, %23) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %27 = "emitc.mul"(%21, %5) : (index, index) -> index
      %28 = "emitc.load"(%20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.subscript"(%4, %27) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %30 = "emitc.load"(%29) : (!emitc.lvalue<f32>) -> f32
      %31 = "emitc.call_opaque"(%28, %30, %26, %23) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%20, %31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %32 = "emitc.load"(%20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %34 = "emitc.apply"(%33) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%34, %32, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x1_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%23 : index):
      %24 = "emitc.mul"(%23, %3) : (index, index) -> index
      %25 = "emitc.constant"() {value = 2 : index} : () -> index
      %26 = "emitc.subscript"(%2, %24) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %28 = "emitc.call_opaque"(%27, %25) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.mul"(%23, %5) : (index, index) -> index
      %30 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %31 = "emitc.subscript"(%4, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %32 = "emitc.load"(%31) : (!emitc.lvalue<f32>) -> f32
      %33 = "emitc.call_opaque"(%30, %32, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%22, %33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %34 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %35 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %36 = "emitc.apply"(%35) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%36, %34, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x2_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%21, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%23 : index):
      %24 = "emitc.mul"(%23, %3) : (index, index) -> index
      %25 = "emitc.constant"() {value = 2 : index} : () -> index
      %26 = "emitc.subscript"(%2, %24) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %28 = "emitc.call_opaque"(%27, %25) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.mul"(%23, %5) : (index, index) -> index
      %30 = "emitc.add"(%29, %11) : (index, index) -> index
      %31 = "emitc.load"(%21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %32 = "emitc.subscript"(%4, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %33 = "emitc.load"(%32) : (!emitc.lvalue<f32>) -> f32
      %34 = "emitc.call_opaque"(%31, %33, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %36 = "emitc.subscript"(%4, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.load"(%36) : (!emitc.lvalue<f32>) -> f32
      %38 = "emitc.call_opaque"(%35, %37, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%21, %34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%22, %38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %39 = "emitc.load"(%21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %40 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %42 = "emitc.apply"(%41) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%42, %39, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %43 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %44 = "emitc.apply"(%43) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%44, %40, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x2_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%27 : index):
      %28 = "emitc.mul"(%27, %3) : (index, index) -> index
      %29 = "emitc.constant"() {value = 2 : index} : () -> index
      %30 = "emitc.subscript"(%2, %28) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %32 = "emitc.call_opaque"(%31, %29) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %33 = "emitc.mul"(%27, %5) : (index, index) -> index
      %34 = "emitc.add"(%33, %11) : (index, index) -> index
      %35 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %36 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.load"(%36) : (!emitc.lvalue<f32>) -> f32
      %38 = "emitc.call_opaque"(%35, %37, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %40 = "emitc.subscript"(%4, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %41 = "emitc.load"(%40) : (!emitc.lvalue<f32>) -> f32
      %42 = "emitc.call_opaque"(%39, %41, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%25, %38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %43 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %44 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %46 = "emitc.apply"(%45) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%46, %43, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %47 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %48 = "emitc.apply"(%47) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%48, %44, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x3_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %23 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%23, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%25 : index):
      %26 = "emitc.mul"(%25, %3) : (index, index) -> index
      %27 = "emitc.constant"() {value = 2 : index} : () -> index
      %28 = "emitc.subscript"(%2, %26) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %29 = "emitc.apply"(%28) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %30 = "emitc.call_opaque"(%29, %27) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %31 = "emitc.mul"(%25, %5) : (index, index) -> index
      %32 = "emitc.add"(%31, %11) : (index, index) -> index
      %33 = "emitc.add"(%31, %12) : (index, index) -> index
      %34 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.subscript"(%4, %31) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %36 = "emitc.load"(%35) : (!emitc.lvalue<f32>) -> f32
      %37 = "emitc.call_opaque"(%34, %36, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.subscript"(%4, %32) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %40 = "emitc.load"(%39) : (!emitc.lvalue<f32>) -> f32
      %41 = "emitc.call_opaque"(%38, %40, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %43 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %44 = "emitc.load"(%43) : (!emitc.lvalue<f32>) -> f32
      %45 = "emitc.call_opaque"(%42, %44, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%22, %37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%23, %41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%24, %45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %46 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %47 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %48 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %49 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %50 = "emitc.apply"(%49) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%50, %46, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %51 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %52 = "emitc.apply"(%51) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%52, %47, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %53 = "emitc.mul"(%12, %8) : (index, index) -> index
    %54 = "emitc.subscript"(%7, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %55 = "emitc.apply"(%54) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%55, %48, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x3_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%32 : index):
      %33 = "emitc.mul"(%32, %3) : (index, index) -> index
      %34 = "emitc.constant"() {value = 2 : index} : () -> index
      %35 = "emitc.subscript"(%2, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %36 = "emitc.apply"(%35) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %37 = "emitc.call_opaque"(%36, %34) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.mul"(%32, %5) : (index, index) -> index
      %39 = "emitc.add"(%38, %11) : (index, index) -> index
      %40 = "emitc.add"(%38, %12) : (index, index) -> index
      %41 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %43 = "emitc.load"(%42) : (!emitc.lvalue<f32>) -> f32
      %44 = "emitc.call_opaque"(%41, %43, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %46 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %47 = "emitc.load"(%46) : (!emitc.lvalue<f32>) -> f32
      %48 = "emitc.call_opaque"(%45, %47, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%29, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %53 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %54 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %55 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %56 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %57 = "emitc.apply"(%56) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%57, %53, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %58 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %59 = "emitc.apply"(%58) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%59, %54, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %60 = "emitc.mul"(%12, %8) : (index, index) -> index
    %61 = "emitc.subscript"(%7, %60) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %62 = "emitc.apply"(%61) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%62, %55, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x4_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%23, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%27 : index):
      %28 = "emitc.mul"(%27, %3) : (index, index) -> index
      %29 = "emitc.constant"() {value = 2 : index} : () -> index
      %30 = "emitc.subscript"(%2, %28) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %32 = "emitc.call_opaque"(%31, %29) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %33 = "emitc.mul"(%27, %5) : (index, index) -> index
      %34 = "emitc.add"(%33, %11) : (index, index) -> index
      %35 = "emitc.add"(%33, %12) : (index, index) -> index
      %36 = "emitc.add"(%33, %13) : (index, index) -> index
      %37 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %39 = "emitc.load"(%38) : (!emitc.lvalue<f32>) -> f32
      %40 = "emitc.call_opaque"(%37, %39, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.subscript"(%4, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %43 = "emitc.load"(%42) : (!emitc.lvalue<f32>) -> f32
      %44 = "emitc.call_opaque"(%41, %43, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %46 = "emitc.subscript"(%4, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %47 = "emitc.load"(%46) : (!emitc.lvalue<f32>) -> f32
      %48 = "emitc.call_opaque"(%45, %47, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%23, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%24, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%25, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %53 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %54 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %55 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %56 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %57 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %58 = "emitc.apply"(%57) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%58, %53, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %59 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %60 = "emitc.apply"(%59) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%60, %54, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %61 = "emitc.mul"(%12, %8) : (index, index) -> index
    %62 = "emitc.subscript"(%7, %61) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %63 = "emitc.apply"(%62) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%63, %55, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %64 = "emitc.mul"(%13, %8) : (index, index) -> index
    %65 = "emitc.subscript"(%7, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %66 = "emitc.apply"(%65) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%66, %56, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x4_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%33, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %34 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%34, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %35 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%35, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %36 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%36, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%37 : index):
      %38 = "emitc.mul"(%37, %3) : (index, index) -> index
      %39 = "emitc.constant"() {value = 2 : index} : () -> index
      %40 = "emitc.subscript"(%2, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %41 = "emitc.apply"(%40) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %42 = "emitc.call_opaque"(%41, %39) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %43 = "emitc.mul"(%37, %5) : (index, index) -> index
      %44 = "emitc.add"(%43, %11) : (index, index) -> index
      %45 = "emitc.add"(%43, %12) : (index, index) -> index
      %46 = "emitc.add"(%43, %13) : (index, index) -> index
      %47 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %49 = "emitc.load"(%48) : (!emitc.lvalue<f32>) -> f32
      %50 = "emitc.call_opaque"(%47, %49, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %53 = "emitc.load"(%52) : (!emitc.lvalue<f32>) -> f32
      %54 = "emitc.call_opaque"(%51, %53, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.load"(%35) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %57 = "emitc.load"(%56) : (!emitc.lvalue<f32>) -> f32
      %58 = "emitc.call_opaque"(%55, %57, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.load"(%36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %46) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%33, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%34, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%35, %58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%36, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %63 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %64 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.load"(%35) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %66 = "emitc.load"(%36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %67 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %68 = "emitc.apply"(%67) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%68, %63, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %69 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %70 = "emitc.apply"(%69) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%70, %64, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %71 = "emitc.mul"(%12, %8) : (index, index) -> index
    %72 = "emitc.subscript"(%7, %71) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %73 = "emitc.apply"(%72) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%73, %65, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %74 = "emitc.mul"(%13, %8) : (index, index) -> index
    %75 = "emitc.subscript"(%7, %74) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %76 = "emitc.apply"(%75) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%76, %66, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x5_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%29 : index):
      %30 = "emitc.mul"(%29, %3) : (index, index) -> index
      %31 = "emitc.constant"() {value = 2 : index} : () -> index
      %32 = "emitc.subscript"(%2, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %33 = "emitc.apply"(%32) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %34 = "emitc.call_opaque"(%33, %31) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.mul"(%29, %5) : (index, index) -> index
      %36 = "emitc.add"(%35, %11) : (index, index) -> index
      %37 = "emitc.add"(%35, %12) : (index, index) -> index
      %38 = "emitc.add"(%35, %13) : (index, index) -> index
      %39 = "emitc.add"(%35, %14) : (index, index) -> index
      %40 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.subscript"(%4, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %42 = "emitc.load"(%41) : (!emitc.lvalue<f32>) -> f32
      %43 = "emitc.call_opaque"(%40, %42, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %44 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.subscript"(%4, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %46 = "emitc.load"(%45) : (!emitc.lvalue<f32>) -> f32
      %47 = "emitc.call_opaque"(%44, %46, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.subscript"(%4, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %50 = "emitc.load"(%49) : (!emitc.lvalue<f32>) -> f32
      %51 = "emitc.call_opaque"(%48, %50, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %54 = "emitc.load"(%53) : (!emitc.lvalue<f32>) -> f32
      %55 = "emitc.call_opaque"(%52, %54, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %58 = "emitc.load"(%57) : (!emitc.lvalue<f32>) -> f32
      %59 = "emitc.call_opaque"(%56, %58, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%24, %43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%25, %47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %59) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %60 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %61 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %62 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %63 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %64 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %66 = "emitc.apply"(%65) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%66, %60, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %67 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %68 = "emitc.apply"(%67) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%68, %61, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %69 = "emitc.mul"(%12, %8) : (index, index) -> index
    %70 = "emitc.subscript"(%7, %69) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %71 = "emitc.apply"(%70) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%71, %62, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %72 = "emitc.mul"(%13, %8) : (index, index) -> index
    %73 = "emitc.subscript"(%7, %72) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %74 = "emitc.apply"(%73) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%74, %63, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %75 = "emitc.mul"(%14, %8) : (index, index) -> index
    %76 = "emitc.subscript"(%7, %75) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %77 = "emitc.apply"(%76) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%77, %64, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x5_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%37, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %38 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%38, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %39 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%39, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %40 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%40, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %41 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%41, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%42 : index):
      %43 = "emitc.mul"(%42, %3) : (index, index) -> index
      %44 = "emitc.constant"() {value = 2 : index} : () -> index
      %45 = "emitc.subscript"(%2, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %46 = "emitc.apply"(%45) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %47 = "emitc.call_opaque"(%46, %44) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.mul"(%42, %5) : (index, index) -> index
      %49 = "emitc.add"(%48, %11) : (index, index) -> index
      %50 = "emitc.add"(%48, %12) : (index, index) -> index
      %51 = "emitc.add"(%48, %13) : (index, index) -> index
      %52 = "emitc.add"(%48, %14) : (index, index) -> index
      %53 = "emitc.load"(%37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.subscript"(%4, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %55 = "emitc.load"(%54) : (!emitc.lvalue<f32>) -> f32
      %56 = "emitc.call_opaque"(%53, %55, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.load"(%38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.subscript"(%4, %49) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %59 = "emitc.load"(%58) : (!emitc.lvalue<f32>) -> f32
      %60 = "emitc.call_opaque"(%57, %59, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %61 = "emitc.load"(%39) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.subscript"(%4, %50) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %63 = "emitc.load"(%62) : (!emitc.lvalue<f32>) -> f32
      %64 = "emitc.call_opaque"(%61, %63, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %65 = "emitc.load"(%40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %51) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %52) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%37, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%38, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%39, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%40, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%41, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %73 = "emitc.load"(%37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %74 = "emitc.load"(%38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %75 = "emitc.load"(%39) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %76 = "emitc.load"(%40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %77 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %78 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %79 = "emitc.apply"(%78) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%79, %73, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %80 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %81 = "emitc.apply"(%80) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%81, %74, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %82 = "emitc.mul"(%12, %8) : (index, index) -> index
    %83 = "emitc.subscript"(%7, %82) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %84 = "emitc.apply"(%83) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%84, %75, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %85 = "emitc.mul"(%13, %8) : (index, index) -> index
    %86 = "emitc.subscript"(%7, %85) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %87 = "emitc.apply"(%86) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%87, %76, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %88 = "emitc.mul"(%14, %8) : (index, index) -> index
    %89 = "emitc.subscript"(%7, %88) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %77, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x6_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%31 : index):
      %32 = "emitc.mul"(%31, %3) : (index, index) -> index
      %33 = "emitc.constant"() {value = 2 : index} : () -> index
      %34 = "emitc.subscript"(%2, %32) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %36 = "emitc.call_opaque"(%35, %33) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %37 = "emitc.mul"(%31, %5) : (index, index) -> index
      %38 = "emitc.add"(%37, %11) : (index, index) -> index
      %39 = "emitc.add"(%37, %12) : (index, index) -> index
      %40 = "emitc.add"(%37, %13) : (index, index) -> index
      %41 = "emitc.add"(%37, %14) : (index, index) -> index
      %42 = "emitc.add"(%37, %15) : (index, index) -> index
      %43 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %44 = "emitc.subscript"(%4, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %45 = "emitc.load"(%44) : (!emitc.lvalue<f32>) -> f32
      %46 = "emitc.call_opaque"(%43, %45, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %47 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %49 = "emitc.load"(%48) : (!emitc.lvalue<f32>) -> f32
      %50 = "emitc.call_opaque"(%47, %49, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %53 = "emitc.load"(%52) : (!emitc.lvalue<f32>) -> f32
      %54 = "emitc.call_opaque"(%51, %53, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %57 = "emitc.load"(%56) : (!emitc.lvalue<f32>) -> f32
      %58 = "emitc.call_opaque"(%55, %57, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %64 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %65 = "emitc.load"(%64) : (!emitc.lvalue<f32>) -> f32
      %66 = "emitc.call_opaque"(%63, %65, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%25, %46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %67 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %68 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %69 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %70 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %71 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %72 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %73 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %74 = "emitc.apply"(%73) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%74, %67, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %75 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %76 = "emitc.apply"(%75) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%76, %68, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %77 = "emitc.mul"(%12, %8) : (index, index) -> index
    %78 = "emitc.subscript"(%7, %77) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %79 = "emitc.apply"(%78) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%79, %69, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %80 = "emitc.mul"(%13, %8) : (index, index) -> index
    %81 = "emitc.subscript"(%7, %80) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %82 = "emitc.apply"(%81) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%82, %70, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %83 = "emitc.mul"(%14, %8) : (index, index) -> index
    %84 = "emitc.subscript"(%7, %83) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %85 = "emitc.apply"(%84) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%85, %71, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %86 = "emitc.mul"(%15, %8) : (index, index) -> index
    %87 = "emitc.subscript"(%7, %86) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %88 = "emitc.apply"(%87) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%88, %72, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x6_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%41, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %42 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%42, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %43 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%43, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %44 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%44, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %45 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%45, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %46 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%46, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%47 : index):
      %48 = "emitc.mul"(%47, %3) : (index, index) -> index
      %49 = "emitc.constant"() {value = 2 : index} : () -> index
      %50 = "emitc.subscript"(%2, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.apply"(%50) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %52 = "emitc.call_opaque"(%51, %49) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.mul"(%47, %5) : (index, index) -> index
      %54 = "emitc.add"(%53, %11) : (index, index) -> index
      %55 = "emitc.add"(%53, %12) : (index, index) -> index
      %56 = "emitc.add"(%53, %13) : (index, index) -> index
      %57 = "emitc.add"(%53, %14) : (index, index) -> index
      %58 = "emitc.add"(%53, %15) : (index, index) -> index
      %59 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.load"(%42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %64 = "emitc.subscript"(%4, %54) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %65 = "emitc.load"(%64) : (!emitc.lvalue<f32>) -> f32
      %66 = "emitc.call_opaque"(%63, %65, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %67 = "emitc.load"(%43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %68 = "emitc.subscript"(%4, %55) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %69 = "emitc.load"(%68) : (!emitc.lvalue<f32>) -> f32
      %70 = "emitc.call_opaque"(%67, %69, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %71 = "emitc.load"(%44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %72 = "emitc.subscript"(%4, %56) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %73 = "emitc.load"(%72) : (!emitc.lvalue<f32>) -> f32
      %74 = "emitc.call_opaque"(%71, %73, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %75 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %76 = "emitc.subscript"(%4, %57) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %77 = "emitc.load"(%76) : (!emitc.lvalue<f32>) -> f32
      %78 = "emitc.call_opaque"(%75, %77, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %79 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %80 = "emitc.subscript"(%4, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %81 = "emitc.load"(%80) : (!emitc.lvalue<f32>) -> f32
      %82 = "emitc.call_opaque"(%79, %81, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%41, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%42, %66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%43, %70) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%44, %74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%45, %78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%46, %82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %83 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %84 = "emitc.load"(%42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %85 = "emitc.load"(%43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %86 = "emitc.load"(%44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %87 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %88 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %89 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %83, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %92 = "emitc.apply"(%91) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%92, %84, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %93 = "emitc.mul"(%12, %8) : (index, index) -> index
    %94 = "emitc.subscript"(%7, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%95, %85, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %96 = "emitc.mul"(%13, %8) : (index, index) -> index
    %97 = "emitc.subscript"(%7, %96) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %98 = "emitc.apply"(%97) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%98, %86, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %99 = "emitc.mul"(%14, %8) : (index, index) -> index
    %100 = "emitc.subscript"(%7, %99) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %87, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.mul"(%15, %8) : (index, index) -> index
    %103 = "emitc.subscript"(%7, %102) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %104 = "emitc.apply"(%103) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%104, %88, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x7_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %32 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%32, %25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%33 : index):
      %34 = "emitc.mul"(%33, %3) : (index, index) -> index
      %35 = "emitc.constant"() {value = 2 : index} : () -> index
      %36 = "emitc.subscript"(%2, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %38 = "emitc.call_opaque"(%37, %35) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.mul"(%33, %5) : (index, index) -> index
      %40 = "emitc.add"(%39, %11) : (index, index) -> index
      %41 = "emitc.add"(%39, %12) : (index, index) -> index
      %42 = "emitc.add"(%39, %13) : (index, index) -> index
      %43 = "emitc.add"(%39, %14) : (index, index) -> index
      %44 = "emitc.add"(%39, %15) : (index, index) -> index
      %45 = "emitc.add"(%39, %16) : (index, index) -> index
      %46 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %47 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %48 = "emitc.load"(%47) : (!emitc.lvalue<f32>) -> f32
      %49 = "emitc.call_opaque"(%46, %48, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %52 = "emitc.load"(%51) : (!emitc.lvalue<f32>) -> f32
      %53 = "emitc.call_opaque"(%50, %52, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %56 = "emitc.load"(%55) : (!emitc.lvalue<f32>) -> f32
      %57 = "emitc.call_opaque"(%54, %56, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %60 = "emitc.load"(%59) : (!emitc.lvalue<f32>) -> f32
      %61 = "emitc.call_opaque"(%58, %60, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %64 = "emitc.load"(%63) : (!emitc.lvalue<f32>) -> f32
      %65 = "emitc.call_opaque"(%62, %64, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %67 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %68 = "emitc.load"(%67) : (!emitc.lvalue<f32>) -> f32
      %69 = "emitc.call_opaque"(%66, %68, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %71 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %72 = "emitc.load"(%71) : (!emitc.lvalue<f32>) -> f32
      %73 = "emitc.call_opaque"(%70, %72, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%26, %49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %57) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %61) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %65) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %69) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%32, %73) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %74 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %75 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %76 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %77 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %78 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %79 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %80 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %81 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %82 = "emitc.apply"(%81) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%82, %74, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %83 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %84 = "emitc.apply"(%83) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%84, %75, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %85 = "emitc.mul"(%12, %8) : (index, index) -> index
    %86 = "emitc.subscript"(%7, %85) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %87 = "emitc.apply"(%86) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%87, %76, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %88 = "emitc.mul"(%13, %8) : (index, index) -> index
    %89 = "emitc.subscript"(%7, %88) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %77, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.mul"(%14, %8) : (index, index) -> index
    %92 = "emitc.subscript"(%7, %91) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %93 = "emitc.apply"(%92) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%93, %78, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %94 = "emitc.mul"(%15, %8) : (index, index) -> index
    %95 = "emitc.subscript"(%7, %94) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %96 = "emitc.apply"(%95) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%96, %79, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %97 = "emitc.mul"(%16, %8) : (index, index) -> index
    %98 = "emitc.subscript"(%7, %97) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %99 = "emitc.apply"(%98) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%99, %80, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x7_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.mul"(%16, %8) : (index, index) -> index
    %42 = "emitc.subscript"(%7, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %43 = "emitc.apply"(%42) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %44 = "emitc.call_opaque"(%43, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%45, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %46 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%46, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %47 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%47, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %48 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%48, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %49 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%49, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %50 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%50, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %51 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%51, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%52 : index):
      %53 = "emitc.mul"(%52, %3) : (index, index) -> index
      %54 = "emitc.constant"() {value = 2 : index} : () -> index
      %55 = "emitc.subscript"(%2, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %56 = "emitc.apply"(%55) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %57 = "emitc.call_opaque"(%56, %54) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.mul"(%52, %5) : (index, index) -> index
      %59 = "emitc.add"(%58, %11) : (index, index) -> index
      %60 = "emitc.add"(%58, %12) : (index, index) -> index
      %61 = "emitc.add"(%58, %13) : (index, index) -> index
      %62 = "emitc.add"(%58, %14) : (index, index) -> index
      %63 = "emitc.add"(%58, %15) : (index, index) -> index
      %64 = "emitc.add"(%58, %16) : (index, index) -> index
      %65 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %59) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %73 = "emitc.load"(%47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %74 = "emitc.subscript"(%4, %60) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %75 = "emitc.load"(%74) : (!emitc.lvalue<f32>) -> f32
      %76 = "emitc.call_opaque"(%73, %75, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %77 = "emitc.load"(%48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %78 = "emitc.subscript"(%4, %61) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %79 = "emitc.load"(%78) : (!emitc.lvalue<f32>) -> f32
      %80 = "emitc.call_opaque"(%77, %79, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %81 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %82 = "emitc.subscript"(%4, %62) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %83 = "emitc.load"(%82) : (!emitc.lvalue<f32>) -> f32
      %84 = "emitc.call_opaque"(%81, %83, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %85 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %86 = "emitc.subscript"(%4, %63) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %87 = "emitc.load"(%86) : (!emitc.lvalue<f32>) -> f32
      %88 = "emitc.call_opaque"(%85, %87, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %89 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %90 = "emitc.subscript"(%4, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %91 = "emitc.load"(%90) : (!emitc.lvalue<f32>) -> f32
      %92 = "emitc.call_opaque"(%89, %91, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%45, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%46, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%47, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%48, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%49, %84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%50, %88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%51, %92) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %93 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %94 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %95 = "emitc.load"(%47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %96 = "emitc.load"(%48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %97 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %98 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %99 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %100 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %93, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %103 = "emitc.apply"(%102) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%103, %94, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %104 = "emitc.mul"(%12, %8) : (index, index) -> index
    %105 = "emitc.subscript"(%7, %104) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %106 = "emitc.apply"(%105) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%106, %95, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %107 = "emitc.mul"(%13, %8) : (index, index) -> index
    %108 = "emitc.subscript"(%7, %107) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %109 = "emitc.apply"(%108) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%109, %96, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %110 = "emitc.mul"(%14, %8) : (index, index) -> index
    %111 = "emitc.subscript"(%7, %110) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %112 = "emitc.apply"(%111) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%112, %97, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %113 = "emitc.mul"(%15, %8) : (index, index) -> index
    %114 = "emitc.subscript"(%7, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %115 = "emitc.apply"(%114) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%115, %98, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %116 = "emitc.mul"(%16, %8) : (index, index) -> index
    %117 = "emitc.subscript"(%7, %116) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %118 = "emitc.apply"(%117) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%118, %99, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x8_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %26 = "emitc.call_opaque"(%9, %12) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %32 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%32, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %33 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%33, %25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %34 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%34, %26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%35 : index):
      %36 = "emitc.mul"(%35, %3) : (index, index) -> index
      %37 = "emitc.constant"() {value = 2 : index} : () -> index
      %38 = "emitc.subscript"(%2, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %40 = "emitc.call_opaque"(%39, %37) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.mul"(%35, %5) : (index, index) -> index
      %42 = "emitc.add"(%41, %11) : (index, index) -> index
      %43 = "emitc.add"(%41, %12) : (index, index) -> index
      %44 = "emitc.add"(%41, %13) : (index, index) -> index
      %45 = "emitc.add"(%41, %14) : (index, index) -> index
      %46 = "emitc.add"(%41, %15) : (index, index) -> index
      %47 = "emitc.add"(%41, %16) : (index, index) -> index
      %48 = "emitc.add"(%41, %17) : (index, index) -> index
      %49 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %55 = "emitc.load"(%54) : (!emitc.lvalue<f32>) -> f32
      %56 = "emitc.call_opaque"(%53, %55, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %59 = "emitc.load"(%58) : (!emitc.lvalue<f32>) -> f32
      %60 = "emitc.call_opaque"(%57, %59, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %61 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %63 = "emitc.load"(%62) : (!emitc.lvalue<f32>) -> f32
      %64 = "emitc.call_opaque"(%61, %63, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %65 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %46) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %73 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %74 = "emitc.subscript"(%4, %47) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %75 = "emitc.load"(%74) : (!emitc.lvalue<f32>) -> f32
      %76 = "emitc.call_opaque"(%73, %75, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %77 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %78 = "emitc.subscript"(%4, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %79 = "emitc.load"(%78) : (!emitc.lvalue<f32>) -> f32
      %80 = "emitc.call_opaque"(%77, %79, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%27, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%32, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%33, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%34, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %81 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %82 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %83 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %84 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %85 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %86 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %87 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %88 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %89 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %81, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %92 = "emitc.apply"(%91) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%92, %82, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %93 = "emitc.mul"(%12, %8) : (index, index) -> index
    %94 = "emitc.subscript"(%7, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%95, %83, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %96 = "emitc.mul"(%13, %8) : (index, index) -> index
    %97 = "emitc.subscript"(%7, %96) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %98 = "emitc.apply"(%97) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%98, %84, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %99 = "emitc.mul"(%14, %8) : (index, index) -> index
    %100 = "emitc.subscript"(%7, %99) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %85, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.mul"(%15, %8) : (index, index) -> index
    %103 = "emitc.subscript"(%7, %102) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %104 = "emitc.apply"(%103) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%104, %86, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %105 = "emitc.mul"(%16, %8) : (index, index) -> index
    %106 = "emitc.subscript"(%7, %105) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %107 = "emitc.apply"(%106) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%107, %87, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %108 = "emitc.mul"(%17, %8) : (index, index) -> index
    %109 = "emitc.subscript"(%7, %108) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %110 = "emitc.apply"(%109) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%110, %88, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_2x8_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.mul"(%16, %8) : (index, index) -> index
    %42 = "emitc.subscript"(%7, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %43 = "emitc.apply"(%42) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %44 = "emitc.call_opaque"(%43, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.mul"(%17, %8) : (index, index) -> index
    %46 = "emitc.subscript"(%7, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %47 = "emitc.apply"(%46) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %48 = "emitc.call_opaque"(%47, %12) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %49 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%49, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %50 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%50, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %51 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%51, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %52 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%52, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %53 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%53, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %54 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%54, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %55 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%55, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %56 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%56, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%57 : index):
      %58 = "emitc.mul"(%57, %3) : (index, index) -> index
      %59 = "emitc.constant"() {value = 2 : index} : () -> index
      %60 = "emitc.subscript"(%2, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.apply"(%60) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %62 = "emitc.call_opaque"(%61, %59) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.mul"(%57, %5) : (index, index) -> index
      %64 = "emitc.add"(%63, %11) : (index, index) -> index
      %65 = "emitc.add"(%63, %12) : (index, index) -> index
      %66 = "emitc.add"(%63, %13) : (index, index) -> index
      %67 = "emitc.add"(%63, %14) : (index, index) -> index
      %68 = "emitc.add"(%63, %15) : (index, index) -> index
      %69 = "emitc.add"(%63, %16) : (index, index) -> index
      %70 = "emitc.add"(%63, %17) : (index, index) -> index
      %71 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %72 = "emitc.subscript"(%4, %63) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %73 = "emitc.load"(%72) : (!emitc.lvalue<f32>) -> f32
      %74 = "emitc.call_opaque"(%71, %73, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %75 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %76 = "emitc.subscript"(%4, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %77 = "emitc.load"(%76) : (!emitc.lvalue<f32>) -> f32
      %78 = "emitc.call_opaque"(%75, %77, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %79 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %80 = "emitc.subscript"(%4, %65) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %81 = "emitc.load"(%80) : (!emitc.lvalue<f32>) -> f32
      %82 = "emitc.call_opaque"(%79, %81, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %83 = "emitc.load"(%52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %84 = "emitc.subscript"(%4, %66) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %85 = "emitc.load"(%84) : (!emitc.lvalue<f32>) -> f32
      %86 = "emitc.call_opaque"(%83, %85, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %87 = "emitc.load"(%53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %88 = "emitc.subscript"(%4, %67) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %89 = "emitc.load"(%88) : (!emitc.lvalue<f32>) -> f32
      %90 = "emitc.call_opaque"(%87, %89, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %91 = "emitc.load"(%54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %92 = "emitc.subscript"(%4, %68) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %93 = "emitc.load"(%92) : (!emitc.lvalue<f32>) -> f32
      %94 = "emitc.call_opaque"(%91, %93, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %95 = "emitc.load"(%55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %96 = "emitc.subscript"(%4, %69) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %97 = "emitc.load"(%96) : (!emitc.lvalue<f32>) -> f32
      %98 = "emitc.call_opaque"(%95, %97, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %99 = "emitc.load"(%56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %100 = "emitc.subscript"(%4, %70) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %101 = "emitc.load"(%100) : (!emitc.lvalue<f32>) -> f32
      %102 = "emitc.call_opaque"(%99, %101, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%49, %74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%50, %78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%51, %82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%52, %86) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%53, %90) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%54, %94) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%55, %98) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%56, %102) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %103 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %104 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %105 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %106 = "emitc.load"(%52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %107 = "emitc.load"(%53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %108 = "emitc.load"(%54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %109 = "emitc.load"(%55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %110 = "emitc.load"(%56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %111 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %112 = "emitc.apply"(%111) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%112, %103, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %113 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %114 = "emitc.apply"(%113) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%114, %104, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %115 = "emitc.mul"(%12, %8) : (index, index) -> index
    %116 = "emitc.subscript"(%7, %115) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %117 = "emitc.apply"(%116) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%117, %105, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %118 = "emitc.mul"(%13, %8) : (index, index) -> index
    %119 = "emitc.subscript"(%7, %118) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %120 = "emitc.apply"(%119) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%120, %106, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %121 = "emitc.mul"(%14, %8) : (index, index) -> index
    %122 = "emitc.subscript"(%7, %121) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %123 = "emitc.apply"(%122) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%123, %107, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %124 = "emitc.mul"(%15, %8) : (index, index) -> index
    %125 = "emitc.subscript"(%7, %124) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %126 = "emitc.apply"(%125) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%126, %108, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %127 = "emitc.mul"(%16, %8) : (index, index) -> index
    %128 = "emitc.subscript"(%7, %127) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %129 = "emitc.apply"(%128) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%129, %109, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %130 = "emitc.mul"(%17, %8) : (index, index) -> index
    %131 = "emitc.subscript"(%7, %130) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %132 = "emitc.apply"(%131) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%132, %110, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x1_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%20, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%21 : index):
      %22 = "emitc.mul"(%21, %3) : (index, index) -> index
      %23 = "emitc.constant"() {value = 3 : index} : () -> index
      %24 = "emitc.subscript"(%2, %22) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %25 = "emitc.apply"(%24) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %26 = "emitc.call_opaque"(%25, %23) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %27 = "emitc.mul"(%21, %5) : (index, index) -> index
      %28 = "emitc.load"(%20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.subscript"(%4, %27) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %30 = "emitc.load"(%29) : (!emitc.lvalue<f32>) -> f32
      %31 = "emitc.call_opaque"(%28, %30, %26, %23) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%20, %31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %32 = "emitc.load"(%20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %34 = "emitc.apply"(%33) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%34, %32, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x1_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%23 : index):
      %24 = "emitc.mul"(%23, %3) : (index, index) -> index
      %25 = "emitc.constant"() {value = 3 : index} : () -> index
      %26 = "emitc.subscript"(%2, %24) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %28 = "emitc.call_opaque"(%27, %25) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.mul"(%23, %5) : (index, index) -> index
      %30 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %31 = "emitc.subscript"(%4, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %32 = "emitc.load"(%31) : (!emitc.lvalue<f32>) -> f32
      %33 = "emitc.call_opaque"(%30, %32, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%22, %33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %34 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %35 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %36 = "emitc.apply"(%35) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%36, %34, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x2_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%21, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%23 : index):
      %24 = "emitc.mul"(%23, %3) : (index, index) -> index
      %25 = "emitc.constant"() {value = 3 : index} : () -> index
      %26 = "emitc.subscript"(%2, %24) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %28 = "emitc.call_opaque"(%27, %25) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.mul"(%23, %5) : (index, index) -> index
      %30 = "emitc.add"(%29, %11) : (index, index) -> index
      %31 = "emitc.load"(%21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %32 = "emitc.subscript"(%4, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %33 = "emitc.load"(%32) : (!emitc.lvalue<f32>) -> f32
      %34 = "emitc.call_opaque"(%31, %33, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %36 = "emitc.subscript"(%4, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.load"(%36) : (!emitc.lvalue<f32>) -> f32
      %38 = "emitc.call_opaque"(%35, %37, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%21, %34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%22, %38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %39 = "emitc.load"(%21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %40 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %42 = "emitc.apply"(%41) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%42, %39, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %43 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %44 = "emitc.apply"(%43) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%44, %40, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x2_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%27 : index):
      %28 = "emitc.mul"(%27, %3) : (index, index) -> index
      %29 = "emitc.constant"() {value = 3 : index} : () -> index
      %30 = "emitc.subscript"(%2, %28) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %32 = "emitc.call_opaque"(%31, %29) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %33 = "emitc.mul"(%27, %5) : (index, index) -> index
      %34 = "emitc.add"(%33, %11) : (index, index) -> index
      %35 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %36 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.load"(%36) : (!emitc.lvalue<f32>) -> f32
      %38 = "emitc.call_opaque"(%35, %37, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %40 = "emitc.subscript"(%4, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %41 = "emitc.load"(%40) : (!emitc.lvalue<f32>) -> f32
      %42 = "emitc.call_opaque"(%39, %41, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%25, %38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %43 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %44 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %46 = "emitc.apply"(%45) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%46, %43, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %47 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %48 = "emitc.apply"(%47) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%48, %44, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x3_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %23 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%23, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%25 : index):
      %26 = "emitc.mul"(%25, %3) : (index, index) -> index
      %27 = "emitc.constant"() {value = 3 : index} : () -> index
      %28 = "emitc.subscript"(%2, %26) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %29 = "emitc.apply"(%28) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %30 = "emitc.call_opaque"(%29, %27) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %31 = "emitc.mul"(%25, %5) : (index, index) -> index
      %32 = "emitc.add"(%31, %11) : (index, index) -> index
      %33 = "emitc.add"(%31, %12) : (index, index) -> index
      %34 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.subscript"(%4, %31) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %36 = "emitc.load"(%35) : (!emitc.lvalue<f32>) -> f32
      %37 = "emitc.call_opaque"(%34, %36, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.subscript"(%4, %32) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %40 = "emitc.load"(%39) : (!emitc.lvalue<f32>) -> f32
      %41 = "emitc.call_opaque"(%38, %40, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %43 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %44 = "emitc.load"(%43) : (!emitc.lvalue<f32>) -> f32
      %45 = "emitc.call_opaque"(%42, %44, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%22, %37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%23, %41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%24, %45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %46 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %47 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %48 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %49 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %50 = "emitc.apply"(%49) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%50, %46, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %51 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %52 = "emitc.apply"(%51) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%52, %47, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %53 = "emitc.mul"(%12, %8) : (index, index) -> index
    %54 = "emitc.subscript"(%7, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %55 = "emitc.apply"(%54) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%55, %48, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x3_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%32 : index):
      %33 = "emitc.mul"(%32, %3) : (index, index) -> index
      %34 = "emitc.constant"() {value = 3 : index} : () -> index
      %35 = "emitc.subscript"(%2, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %36 = "emitc.apply"(%35) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %37 = "emitc.call_opaque"(%36, %34) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.mul"(%32, %5) : (index, index) -> index
      %39 = "emitc.add"(%38, %11) : (index, index) -> index
      %40 = "emitc.add"(%38, %12) : (index, index) -> index
      %41 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %43 = "emitc.load"(%42) : (!emitc.lvalue<f32>) -> f32
      %44 = "emitc.call_opaque"(%41, %43, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %46 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %47 = "emitc.load"(%46) : (!emitc.lvalue<f32>) -> f32
      %48 = "emitc.call_opaque"(%45, %47, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%29, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %53 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %54 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %55 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %56 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %57 = "emitc.apply"(%56) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%57, %53, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %58 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %59 = "emitc.apply"(%58) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%59, %54, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %60 = "emitc.mul"(%12, %8) : (index, index) -> index
    %61 = "emitc.subscript"(%7, %60) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %62 = "emitc.apply"(%61) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%62, %55, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x4_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%23, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%27 : index):
      %28 = "emitc.mul"(%27, %3) : (index, index) -> index
      %29 = "emitc.constant"() {value = 3 : index} : () -> index
      %30 = "emitc.subscript"(%2, %28) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %32 = "emitc.call_opaque"(%31, %29) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %33 = "emitc.mul"(%27, %5) : (index, index) -> index
      %34 = "emitc.add"(%33, %11) : (index, index) -> index
      %35 = "emitc.add"(%33, %12) : (index, index) -> index
      %36 = "emitc.add"(%33, %13) : (index, index) -> index
      %37 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %39 = "emitc.load"(%38) : (!emitc.lvalue<f32>) -> f32
      %40 = "emitc.call_opaque"(%37, %39, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.subscript"(%4, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %43 = "emitc.load"(%42) : (!emitc.lvalue<f32>) -> f32
      %44 = "emitc.call_opaque"(%41, %43, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %46 = "emitc.subscript"(%4, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %47 = "emitc.load"(%46) : (!emitc.lvalue<f32>) -> f32
      %48 = "emitc.call_opaque"(%45, %47, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%23, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%24, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%25, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %53 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %54 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %55 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %56 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %57 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %58 = "emitc.apply"(%57) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%58, %53, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %59 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %60 = "emitc.apply"(%59) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%60, %54, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %61 = "emitc.mul"(%12, %8) : (index, index) -> index
    %62 = "emitc.subscript"(%7, %61) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %63 = "emitc.apply"(%62) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%63, %55, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %64 = "emitc.mul"(%13, %8) : (index, index) -> index
    %65 = "emitc.subscript"(%7, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %66 = "emitc.apply"(%65) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%66, %56, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x4_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%33, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %34 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%34, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %35 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%35, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %36 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%36, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%37 : index):
      %38 = "emitc.mul"(%37, %3) : (index, index) -> index
      %39 = "emitc.constant"() {value = 3 : index} : () -> index
      %40 = "emitc.subscript"(%2, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %41 = "emitc.apply"(%40) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %42 = "emitc.call_opaque"(%41, %39) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %43 = "emitc.mul"(%37, %5) : (index, index) -> index
      %44 = "emitc.add"(%43, %11) : (index, index) -> index
      %45 = "emitc.add"(%43, %12) : (index, index) -> index
      %46 = "emitc.add"(%43, %13) : (index, index) -> index
      %47 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %49 = "emitc.load"(%48) : (!emitc.lvalue<f32>) -> f32
      %50 = "emitc.call_opaque"(%47, %49, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %53 = "emitc.load"(%52) : (!emitc.lvalue<f32>) -> f32
      %54 = "emitc.call_opaque"(%51, %53, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.load"(%35) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %57 = "emitc.load"(%56) : (!emitc.lvalue<f32>) -> f32
      %58 = "emitc.call_opaque"(%55, %57, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.load"(%36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %46) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%33, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%34, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%35, %58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%36, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %63 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %64 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.load"(%35) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %66 = "emitc.load"(%36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %67 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %68 = "emitc.apply"(%67) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%68, %63, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %69 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %70 = "emitc.apply"(%69) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%70, %64, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %71 = "emitc.mul"(%12, %8) : (index, index) -> index
    %72 = "emitc.subscript"(%7, %71) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %73 = "emitc.apply"(%72) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%73, %65, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %74 = "emitc.mul"(%13, %8) : (index, index) -> index
    %75 = "emitc.subscript"(%7, %74) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %76 = "emitc.apply"(%75) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%76, %66, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x5_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%29 : index):
      %30 = "emitc.mul"(%29, %3) : (index, index) -> index
      %31 = "emitc.constant"() {value = 3 : index} : () -> index
      %32 = "emitc.subscript"(%2, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %33 = "emitc.apply"(%32) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %34 = "emitc.call_opaque"(%33, %31) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.mul"(%29, %5) : (index, index) -> index
      %36 = "emitc.add"(%35, %11) : (index, index) -> index
      %37 = "emitc.add"(%35, %12) : (index, index) -> index
      %38 = "emitc.add"(%35, %13) : (index, index) -> index
      %39 = "emitc.add"(%35, %14) : (index, index) -> index
      %40 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.subscript"(%4, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %42 = "emitc.load"(%41) : (!emitc.lvalue<f32>) -> f32
      %43 = "emitc.call_opaque"(%40, %42, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %44 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.subscript"(%4, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %46 = "emitc.load"(%45) : (!emitc.lvalue<f32>) -> f32
      %47 = "emitc.call_opaque"(%44, %46, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.subscript"(%4, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %50 = "emitc.load"(%49) : (!emitc.lvalue<f32>) -> f32
      %51 = "emitc.call_opaque"(%48, %50, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %54 = "emitc.load"(%53) : (!emitc.lvalue<f32>) -> f32
      %55 = "emitc.call_opaque"(%52, %54, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %58 = "emitc.load"(%57) : (!emitc.lvalue<f32>) -> f32
      %59 = "emitc.call_opaque"(%56, %58, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%24, %43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%25, %47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %59) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %60 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %61 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %62 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %63 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %64 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %66 = "emitc.apply"(%65) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%66, %60, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %67 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %68 = "emitc.apply"(%67) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%68, %61, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %69 = "emitc.mul"(%12, %8) : (index, index) -> index
    %70 = "emitc.subscript"(%7, %69) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %71 = "emitc.apply"(%70) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%71, %62, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %72 = "emitc.mul"(%13, %8) : (index, index) -> index
    %73 = "emitc.subscript"(%7, %72) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %74 = "emitc.apply"(%73) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%74, %63, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %75 = "emitc.mul"(%14, %8) : (index, index) -> index
    %76 = "emitc.subscript"(%7, %75) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %77 = "emitc.apply"(%76) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%77, %64, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x5_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%37, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %38 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%38, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %39 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%39, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %40 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%40, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %41 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%41, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%42 : index):
      %43 = "emitc.mul"(%42, %3) : (index, index) -> index
      %44 = "emitc.constant"() {value = 3 : index} : () -> index
      %45 = "emitc.subscript"(%2, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %46 = "emitc.apply"(%45) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %47 = "emitc.call_opaque"(%46, %44) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.mul"(%42, %5) : (index, index) -> index
      %49 = "emitc.add"(%48, %11) : (index, index) -> index
      %50 = "emitc.add"(%48, %12) : (index, index) -> index
      %51 = "emitc.add"(%48, %13) : (index, index) -> index
      %52 = "emitc.add"(%48, %14) : (index, index) -> index
      %53 = "emitc.load"(%37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.subscript"(%4, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %55 = "emitc.load"(%54) : (!emitc.lvalue<f32>) -> f32
      %56 = "emitc.call_opaque"(%53, %55, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.load"(%38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.subscript"(%4, %49) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %59 = "emitc.load"(%58) : (!emitc.lvalue<f32>) -> f32
      %60 = "emitc.call_opaque"(%57, %59, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %61 = "emitc.load"(%39) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.subscript"(%4, %50) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %63 = "emitc.load"(%62) : (!emitc.lvalue<f32>) -> f32
      %64 = "emitc.call_opaque"(%61, %63, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %65 = "emitc.load"(%40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %51) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %52) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%37, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%38, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%39, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%40, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%41, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %73 = "emitc.load"(%37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %74 = "emitc.load"(%38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %75 = "emitc.load"(%39) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %76 = "emitc.load"(%40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %77 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %78 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %79 = "emitc.apply"(%78) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%79, %73, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %80 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %81 = "emitc.apply"(%80) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%81, %74, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %82 = "emitc.mul"(%12, %8) : (index, index) -> index
    %83 = "emitc.subscript"(%7, %82) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %84 = "emitc.apply"(%83) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%84, %75, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %85 = "emitc.mul"(%13, %8) : (index, index) -> index
    %86 = "emitc.subscript"(%7, %85) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %87 = "emitc.apply"(%86) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%87, %76, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %88 = "emitc.mul"(%14, %8) : (index, index) -> index
    %89 = "emitc.subscript"(%7, %88) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %77, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x6_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%31 : index):
      %32 = "emitc.mul"(%31, %3) : (index, index) -> index
      %33 = "emitc.constant"() {value = 3 : index} : () -> index
      %34 = "emitc.subscript"(%2, %32) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %36 = "emitc.call_opaque"(%35, %33) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %37 = "emitc.mul"(%31, %5) : (index, index) -> index
      %38 = "emitc.add"(%37, %11) : (index, index) -> index
      %39 = "emitc.add"(%37, %12) : (index, index) -> index
      %40 = "emitc.add"(%37, %13) : (index, index) -> index
      %41 = "emitc.add"(%37, %14) : (index, index) -> index
      %42 = "emitc.add"(%37, %15) : (index, index) -> index
      %43 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %44 = "emitc.subscript"(%4, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %45 = "emitc.load"(%44) : (!emitc.lvalue<f32>) -> f32
      %46 = "emitc.call_opaque"(%43, %45, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %47 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %49 = "emitc.load"(%48) : (!emitc.lvalue<f32>) -> f32
      %50 = "emitc.call_opaque"(%47, %49, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %53 = "emitc.load"(%52) : (!emitc.lvalue<f32>) -> f32
      %54 = "emitc.call_opaque"(%51, %53, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %57 = "emitc.load"(%56) : (!emitc.lvalue<f32>) -> f32
      %58 = "emitc.call_opaque"(%55, %57, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %64 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %65 = "emitc.load"(%64) : (!emitc.lvalue<f32>) -> f32
      %66 = "emitc.call_opaque"(%63, %65, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%25, %46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %67 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %68 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %69 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %70 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %71 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %72 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %73 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %74 = "emitc.apply"(%73) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%74, %67, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %75 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %76 = "emitc.apply"(%75) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%76, %68, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %77 = "emitc.mul"(%12, %8) : (index, index) -> index
    %78 = "emitc.subscript"(%7, %77) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %79 = "emitc.apply"(%78) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%79, %69, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %80 = "emitc.mul"(%13, %8) : (index, index) -> index
    %81 = "emitc.subscript"(%7, %80) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %82 = "emitc.apply"(%81) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%82, %70, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %83 = "emitc.mul"(%14, %8) : (index, index) -> index
    %84 = "emitc.subscript"(%7, %83) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %85 = "emitc.apply"(%84) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%85, %71, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %86 = "emitc.mul"(%15, %8) : (index, index) -> index
    %87 = "emitc.subscript"(%7, %86) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %88 = "emitc.apply"(%87) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%88, %72, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x6_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%41, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %42 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%42, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %43 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%43, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %44 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%44, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %45 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%45, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %46 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%46, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%47 : index):
      %48 = "emitc.mul"(%47, %3) : (index, index) -> index
      %49 = "emitc.constant"() {value = 3 : index} : () -> index
      %50 = "emitc.subscript"(%2, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.apply"(%50) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %52 = "emitc.call_opaque"(%51, %49) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.mul"(%47, %5) : (index, index) -> index
      %54 = "emitc.add"(%53, %11) : (index, index) -> index
      %55 = "emitc.add"(%53, %12) : (index, index) -> index
      %56 = "emitc.add"(%53, %13) : (index, index) -> index
      %57 = "emitc.add"(%53, %14) : (index, index) -> index
      %58 = "emitc.add"(%53, %15) : (index, index) -> index
      %59 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.load"(%42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %64 = "emitc.subscript"(%4, %54) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %65 = "emitc.load"(%64) : (!emitc.lvalue<f32>) -> f32
      %66 = "emitc.call_opaque"(%63, %65, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %67 = "emitc.load"(%43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %68 = "emitc.subscript"(%4, %55) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %69 = "emitc.load"(%68) : (!emitc.lvalue<f32>) -> f32
      %70 = "emitc.call_opaque"(%67, %69, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %71 = "emitc.load"(%44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %72 = "emitc.subscript"(%4, %56) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %73 = "emitc.load"(%72) : (!emitc.lvalue<f32>) -> f32
      %74 = "emitc.call_opaque"(%71, %73, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %75 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %76 = "emitc.subscript"(%4, %57) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %77 = "emitc.load"(%76) : (!emitc.lvalue<f32>) -> f32
      %78 = "emitc.call_opaque"(%75, %77, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %79 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %80 = "emitc.subscript"(%4, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %81 = "emitc.load"(%80) : (!emitc.lvalue<f32>) -> f32
      %82 = "emitc.call_opaque"(%79, %81, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%41, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%42, %66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%43, %70) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%44, %74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%45, %78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%46, %82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %83 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %84 = "emitc.load"(%42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %85 = "emitc.load"(%43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %86 = "emitc.load"(%44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %87 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %88 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %89 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %83, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %92 = "emitc.apply"(%91) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%92, %84, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %93 = "emitc.mul"(%12, %8) : (index, index) -> index
    %94 = "emitc.subscript"(%7, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%95, %85, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %96 = "emitc.mul"(%13, %8) : (index, index) -> index
    %97 = "emitc.subscript"(%7, %96) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %98 = "emitc.apply"(%97) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%98, %86, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %99 = "emitc.mul"(%14, %8) : (index, index) -> index
    %100 = "emitc.subscript"(%7, %99) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %87, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.mul"(%15, %8) : (index, index) -> index
    %103 = "emitc.subscript"(%7, %102) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %104 = "emitc.apply"(%103) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%104, %88, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x7_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %32 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%32, %25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%33 : index):
      %34 = "emitc.mul"(%33, %3) : (index, index) -> index
      %35 = "emitc.constant"() {value = 3 : index} : () -> index
      %36 = "emitc.subscript"(%2, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %38 = "emitc.call_opaque"(%37, %35) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.mul"(%33, %5) : (index, index) -> index
      %40 = "emitc.add"(%39, %11) : (index, index) -> index
      %41 = "emitc.add"(%39, %12) : (index, index) -> index
      %42 = "emitc.add"(%39, %13) : (index, index) -> index
      %43 = "emitc.add"(%39, %14) : (index, index) -> index
      %44 = "emitc.add"(%39, %15) : (index, index) -> index
      %45 = "emitc.add"(%39, %16) : (index, index) -> index
      %46 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %47 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %48 = "emitc.load"(%47) : (!emitc.lvalue<f32>) -> f32
      %49 = "emitc.call_opaque"(%46, %48, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %52 = "emitc.load"(%51) : (!emitc.lvalue<f32>) -> f32
      %53 = "emitc.call_opaque"(%50, %52, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %56 = "emitc.load"(%55) : (!emitc.lvalue<f32>) -> f32
      %57 = "emitc.call_opaque"(%54, %56, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %60 = "emitc.load"(%59) : (!emitc.lvalue<f32>) -> f32
      %61 = "emitc.call_opaque"(%58, %60, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %64 = "emitc.load"(%63) : (!emitc.lvalue<f32>) -> f32
      %65 = "emitc.call_opaque"(%62, %64, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %67 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %68 = "emitc.load"(%67) : (!emitc.lvalue<f32>) -> f32
      %69 = "emitc.call_opaque"(%66, %68, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %71 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %72 = "emitc.load"(%71) : (!emitc.lvalue<f32>) -> f32
      %73 = "emitc.call_opaque"(%70, %72, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%26, %49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %57) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %61) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %65) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %69) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%32, %73) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %74 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %75 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %76 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %77 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %78 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %79 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %80 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %81 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %82 = "emitc.apply"(%81) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%82, %74, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %83 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %84 = "emitc.apply"(%83) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%84, %75, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %85 = "emitc.mul"(%12, %8) : (index, index) -> index
    %86 = "emitc.subscript"(%7, %85) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %87 = "emitc.apply"(%86) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%87, %76, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %88 = "emitc.mul"(%13, %8) : (index, index) -> index
    %89 = "emitc.subscript"(%7, %88) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %77, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.mul"(%14, %8) : (index, index) -> index
    %92 = "emitc.subscript"(%7, %91) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %93 = "emitc.apply"(%92) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%93, %78, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %94 = "emitc.mul"(%15, %8) : (index, index) -> index
    %95 = "emitc.subscript"(%7, %94) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %96 = "emitc.apply"(%95) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%96, %79, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %97 = "emitc.mul"(%16, %8) : (index, index) -> index
    %98 = "emitc.subscript"(%7, %97) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %99 = "emitc.apply"(%98) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%99, %80, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x7_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.mul"(%16, %8) : (index, index) -> index
    %42 = "emitc.subscript"(%7, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %43 = "emitc.apply"(%42) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %44 = "emitc.call_opaque"(%43, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%45, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %46 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%46, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %47 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%47, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %48 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%48, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %49 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%49, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %50 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%50, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %51 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%51, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%52 : index):
      %53 = "emitc.mul"(%52, %3) : (index, index) -> index
      %54 = "emitc.constant"() {value = 3 : index} : () -> index
      %55 = "emitc.subscript"(%2, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %56 = "emitc.apply"(%55) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %57 = "emitc.call_opaque"(%56, %54) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.mul"(%52, %5) : (index, index) -> index
      %59 = "emitc.add"(%58, %11) : (index, index) -> index
      %60 = "emitc.add"(%58, %12) : (index, index) -> index
      %61 = "emitc.add"(%58, %13) : (index, index) -> index
      %62 = "emitc.add"(%58, %14) : (index, index) -> index
      %63 = "emitc.add"(%58, %15) : (index, index) -> index
      %64 = "emitc.add"(%58, %16) : (index, index) -> index
      %65 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %59) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %73 = "emitc.load"(%47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %74 = "emitc.subscript"(%4, %60) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %75 = "emitc.load"(%74) : (!emitc.lvalue<f32>) -> f32
      %76 = "emitc.call_opaque"(%73, %75, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %77 = "emitc.load"(%48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %78 = "emitc.subscript"(%4, %61) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %79 = "emitc.load"(%78) : (!emitc.lvalue<f32>) -> f32
      %80 = "emitc.call_opaque"(%77, %79, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %81 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %82 = "emitc.subscript"(%4, %62) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %83 = "emitc.load"(%82) : (!emitc.lvalue<f32>) -> f32
      %84 = "emitc.call_opaque"(%81, %83, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %85 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %86 = "emitc.subscript"(%4, %63) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %87 = "emitc.load"(%86) : (!emitc.lvalue<f32>) -> f32
      %88 = "emitc.call_opaque"(%85, %87, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %89 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %90 = "emitc.subscript"(%4, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %91 = "emitc.load"(%90) : (!emitc.lvalue<f32>) -> f32
      %92 = "emitc.call_opaque"(%89, %91, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%45, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%46, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%47, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%48, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%49, %84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%50, %88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%51, %92) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %93 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %94 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %95 = "emitc.load"(%47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %96 = "emitc.load"(%48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %97 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %98 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %99 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %100 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %93, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %103 = "emitc.apply"(%102) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%103, %94, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %104 = "emitc.mul"(%12, %8) : (index, index) -> index
    %105 = "emitc.subscript"(%7, %104) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %106 = "emitc.apply"(%105) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%106, %95, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %107 = "emitc.mul"(%13, %8) : (index, index) -> index
    %108 = "emitc.subscript"(%7, %107) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %109 = "emitc.apply"(%108) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%109, %96, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %110 = "emitc.mul"(%14, %8) : (index, index) -> index
    %111 = "emitc.subscript"(%7, %110) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %112 = "emitc.apply"(%111) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%112, %97, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %113 = "emitc.mul"(%15, %8) : (index, index) -> index
    %114 = "emitc.subscript"(%7, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %115 = "emitc.apply"(%114) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%115, %98, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %116 = "emitc.mul"(%16, %8) : (index, index) -> index
    %117 = "emitc.subscript"(%7, %116) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %118 = "emitc.apply"(%117) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%118, %99, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x8_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %26 = "emitc.call_opaque"(%9, %13) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %32 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%32, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %33 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%33, %25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %34 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%34, %26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%35 : index):
      %36 = "emitc.mul"(%35, %3) : (index, index) -> index
      %37 = "emitc.constant"() {value = 3 : index} : () -> index
      %38 = "emitc.subscript"(%2, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %40 = "emitc.call_opaque"(%39, %37) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.mul"(%35, %5) : (index, index) -> index
      %42 = "emitc.add"(%41, %11) : (index, index) -> index
      %43 = "emitc.add"(%41, %12) : (index, index) -> index
      %44 = "emitc.add"(%41, %13) : (index, index) -> index
      %45 = "emitc.add"(%41, %14) : (index, index) -> index
      %46 = "emitc.add"(%41, %15) : (index, index) -> index
      %47 = "emitc.add"(%41, %16) : (index, index) -> index
      %48 = "emitc.add"(%41, %17) : (index, index) -> index
      %49 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %55 = "emitc.load"(%54) : (!emitc.lvalue<f32>) -> f32
      %56 = "emitc.call_opaque"(%53, %55, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %59 = "emitc.load"(%58) : (!emitc.lvalue<f32>) -> f32
      %60 = "emitc.call_opaque"(%57, %59, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %61 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %63 = "emitc.load"(%62) : (!emitc.lvalue<f32>) -> f32
      %64 = "emitc.call_opaque"(%61, %63, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %65 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %46) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %73 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %74 = "emitc.subscript"(%4, %47) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %75 = "emitc.load"(%74) : (!emitc.lvalue<f32>) -> f32
      %76 = "emitc.call_opaque"(%73, %75, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %77 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %78 = "emitc.subscript"(%4, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %79 = "emitc.load"(%78) : (!emitc.lvalue<f32>) -> f32
      %80 = "emitc.call_opaque"(%77, %79, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%27, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%32, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%33, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%34, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %81 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %82 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %83 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %84 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %85 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %86 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %87 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %88 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %89 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %81, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %92 = "emitc.apply"(%91) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%92, %82, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %93 = "emitc.mul"(%12, %8) : (index, index) -> index
    %94 = "emitc.subscript"(%7, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%95, %83, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %96 = "emitc.mul"(%13, %8) : (index, index) -> index
    %97 = "emitc.subscript"(%7, %96) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %98 = "emitc.apply"(%97) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%98, %84, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %99 = "emitc.mul"(%14, %8) : (index, index) -> index
    %100 = "emitc.subscript"(%7, %99) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %85, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.mul"(%15, %8) : (index, index) -> index
    %103 = "emitc.subscript"(%7, %102) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %104 = "emitc.apply"(%103) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%104, %86, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %105 = "emitc.mul"(%16, %8) : (index, index) -> index
    %106 = "emitc.subscript"(%7, %105) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %107 = "emitc.apply"(%106) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%107, %87, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %108 = "emitc.mul"(%17, %8) : (index, index) -> index
    %109 = "emitc.subscript"(%7, %108) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %110 = "emitc.apply"(%109) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%110, %88, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_3x8_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.mul"(%16, %8) : (index, index) -> index
    %42 = "emitc.subscript"(%7, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %43 = "emitc.apply"(%42) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %44 = "emitc.call_opaque"(%43, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.mul"(%17, %8) : (index, index) -> index
    %46 = "emitc.subscript"(%7, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %47 = "emitc.apply"(%46) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %48 = "emitc.call_opaque"(%47, %13) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %49 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%49, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %50 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%50, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %51 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%51, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %52 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%52, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %53 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%53, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %54 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%54, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %55 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%55, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %56 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%56, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%57 : index):
      %58 = "emitc.mul"(%57, %3) : (index, index) -> index
      %59 = "emitc.constant"() {value = 3 : index} : () -> index
      %60 = "emitc.subscript"(%2, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.apply"(%60) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %62 = "emitc.call_opaque"(%61, %59) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.mul"(%57, %5) : (index, index) -> index
      %64 = "emitc.add"(%63, %11) : (index, index) -> index
      %65 = "emitc.add"(%63, %12) : (index, index) -> index
      %66 = "emitc.add"(%63, %13) : (index, index) -> index
      %67 = "emitc.add"(%63, %14) : (index, index) -> index
      %68 = "emitc.add"(%63, %15) : (index, index) -> index
      %69 = "emitc.add"(%63, %16) : (index, index) -> index
      %70 = "emitc.add"(%63, %17) : (index, index) -> index
      %71 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %72 = "emitc.subscript"(%4, %63) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %73 = "emitc.load"(%72) : (!emitc.lvalue<f32>) -> f32
      %74 = "emitc.call_opaque"(%71, %73, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %75 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %76 = "emitc.subscript"(%4, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %77 = "emitc.load"(%76) : (!emitc.lvalue<f32>) -> f32
      %78 = "emitc.call_opaque"(%75, %77, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %79 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %80 = "emitc.subscript"(%4, %65) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %81 = "emitc.load"(%80) : (!emitc.lvalue<f32>) -> f32
      %82 = "emitc.call_opaque"(%79, %81, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %83 = "emitc.load"(%52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %84 = "emitc.subscript"(%4, %66) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %85 = "emitc.load"(%84) : (!emitc.lvalue<f32>) -> f32
      %86 = "emitc.call_opaque"(%83, %85, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %87 = "emitc.load"(%53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %88 = "emitc.subscript"(%4, %67) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %89 = "emitc.load"(%88) : (!emitc.lvalue<f32>) -> f32
      %90 = "emitc.call_opaque"(%87, %89, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %91 = "emitc.load"(%54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %92 = "emitc.subscript"(%4, %68) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %93 = "emitc.load"(%92) : (!emitc.lvalue<f32>) -> f32
      %94 = "emitc.call_opaque"(%91, %93, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %95 = "emitc.load"(%55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %96 = "emitc.subscript"(%4, %69) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %97 = "emitc.load"(%96) : (!emitc.lvalue<f32>) -> f32
      %98 = "emitc.call_opaque"(%95, %97, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %99 = "emitc.load"(%56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %100 = "emitc.subscript"(%4, %70) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %101 = "emitc.load"(%100) : (!emitc.lvalue<f32>) -> f32
      %102 = "emitc.call_opaque"(%99, %101, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%49, %74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%50, %78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%51, %82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%52, %86) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%53, %90) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%54, %94) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%55, %98) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%56, %102) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %103 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %104 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %105 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %106 = "emitc.load"(%52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %107 = "emitc.load"(%53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %108 = "emitc.load"(%54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %109 = "emitc.load"(%55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %110 = "emitc.load"(%56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %111 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %112 = "emitc.apply"(%111) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%112, %103, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %113 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %114 = "emitc.apply"(%113) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%114, %104, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %115 = "emitc.mul"(%12, %8) : (index, index) -> index
    %116 = "emitc.subscript"(%7, %115) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %117 = "emitc.apply"(%116) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%117, %105, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %118 = "emitc.mul"(%13, %8) : (index, index) -> index
    %119 = "emitc.subscript"(%7, %118) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %120 = "emitc.apply"(%119) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%120, %106, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %121 = "emitc.mul"(%14, %8) : (index, index) -> index
    %122 = "emitc.subscript"(%7, %121) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %123 = "emitc.apply"(%122) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%123, %107, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %124 = "emitc.mul"(%15, %8) : (index, index) -> index
    %125 = "emitc.subscript"(%7, %124) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %126 = "emitc.apply"(%125) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%126, %108, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %127 = "emitc.mul"(%16, %8) : (index, index) -> index
    %128 = "emitc.subscript"(%7, %127) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %129 = "emitc.apply"(%128) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%129, %109, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %130 = "emitc.mul"(%17, %8) : (index, index) -> index
    %131 = "emitc.subscript"(%7, %130) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %132 = "emitc.apply"(%131) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%132, %110, %13) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x1_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%20, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%21 : index):
      %22 = "emitc.mul"(%21, %3) : (index, index) -> index
      %23 = "emitc.constant"() {value = 4 : index} : () -> index
      %24 = "emitc.subscript"(%2, %22) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %25 = "emitc.apply"(%24) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %26 = "emitc.call_opaque"(%25, %23) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %27 = "emitc.mul"(%21, %5) : (index, index) -> index
      %28 = "emitc.load"(%20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.subscript"(%4, %27) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %30 = "emitc.load"(%29) : (!emitc.lvalue<f32>) -> f32
      %31 = "emitc.call_opaque"(%28, %30, %26, %23) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%20, %31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %32 = "emitc.load"(%20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %34 = "emitc.apply"(%33) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%34, %32, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x1_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%23 : index):
      %24 = "emitc.mul"(%23, %3) : (index, index) -> index
      %25 = "emitc.constant"() {value = 4 : index} : () -> index
      %26 = "emitc.subscript"(%2, %24) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %28 = "emitc.call_opaque"(%27, %25) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.mul"(%23, %5) : (index, index) -> index
      %30 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %31 = "emitc.subscript"(%4, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %32 = "emitc.load"(%31) : (!emitc.lvalue<f32>) -> f32
      %33 = "emitc.call_opaque"(%30, %32, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%22, %33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %34 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %35 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %36 = "emitc.apply"(%35) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%36, %34, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x2_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%21, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%23 : index):
      %24 = "emitc.mul"(%23, %3) : (index, index) -> index
      %25 = "emitc.constant"() {value = 4 : index} : () -> index
      %26 = "emitc.subscript"(%2, %24) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %28 = "emitc.call_opaque"(%27, %25) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %29 = "emitc.mul"(%23, %5) : (index, index) -> index
      %30 = "emitc.add"(%29, %11) : (index, index) -> index
      %31 = "emitc.load"(%21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %32 = "emitc.subscript"(%4, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %33 = "emitc.load"(%32) : (!emitc.lvalue<f32>) -> f32
      %34 = "emitc.call_opaque"(%31, %33, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %36 = "emitc.subscript"(%4, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.load"(%36) : (!emitc.lvalue<f32>) -> f32
      %38 = "emitc.call_opaque"(%35, %37, %28, %25) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%21, %34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%22, %38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %39 = "emitc.load"(%21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %40 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %42 = "emitc.apply"(%41) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%42, %39, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %43 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %44 = "emitc.apply"(%43) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%44, %40, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x2_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%27 : index):
      %28 = "emitc.mul"(%27, %3) : (index, index) -> index
      %29 = "emitc.constant"() {value = 4 : index} : () -> index
      %30 = "emitc.subscript"(%2, %28) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %32 = "emitc.call_opaque"(%31, %29) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %33 = "emitc.mul"(%27, %5) : (index, index) -> index
      %34 = "emitc.add"(%33, %11) : (index, index) -> index
      %35 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %36 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.load"(%36) : (!emitc.lvalue<f32>) -> f32
      %38 = "emitc.call_opaque"(%35, %37, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %40 = "emitc.subscript"(%4, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %41 = "emitc.load"(%40) : (!emitc.lvalue<f32>) -> f32
      %42 = "emitc.call_opaque"(%39, %41, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%25, %38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %43 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %44 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %46 = "emitc.apply"(%45) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%46, %43, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %47 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %48 = "emitc.apply"(%47) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%48, %44, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x3_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%22, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %23 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%23, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%25 : index):
      %26 = "emitc.mul"(%25, %3) : (index, index) -> index
      %27 = "emitc.constant"() {value = 4 : index} : () -> index
      %28 = "emitc.subscript"(%2, %26) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %29 = "emitc.apply"(%28) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %30 = "emitc.call_opaque"(%29, %27) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %31 = "emitc.mul"(%25, %5) : (index, index) -> index
      %32 = "emitc.add"(%31, %11) : (index, index) -> index
      %33 = "emitc.add"(%31, %12) : (index, index) -> index
      %34 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.subscript"(%4, %31) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %36 = "emitc.load"(%35) : (!emitc.lvalue<f32>) -> f32
      %37 = "emitc.call_opaque"(%34, %36, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.subscript"(%4, %32) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %40 = "emitc.load"(%39) : (!emitc.lvalue<f32>) -> f32
      %41 = "emitc.call_opaque"(%38, %40, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %43 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %44 = "emitc.load"(%43) : (!emitc.lvalue<f32>) -> f32
      %45 = "emitc.call_opaque"(%42, %44, %30, %27) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%22, %37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%23, %41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%24, %45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %46 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %47 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %48 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %49 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %50 = "emitc.apply"(%49) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%50, %46, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %51 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %52 = "emitc.apply"(%51) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%52, %47, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %53 = "emitc.mul"(%12, %8) : (index, index) -> index
    %54 = "emitc.subscript"(%7, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %55 = "emitc.apply"(%54) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%55, %48, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x3_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%32 : index):
      %33 = "emitc.mul"(%32, %3) : (index, index) -> index
      %34 = "emitc.constant"() {value = 4 : index} : () -> index
      %35 = "emitc.subscript"(%2, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %36 = "emitc.apply"(%35) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %37 = "emitc.call_opaque"(%36, %34) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.mul"(%32, %5) : (index, index) -> index
      %39 = "emitc.add"(%38, %11) : (index, index) -> index
      %40 = "emitc.add"(%38, %12) : (index, index) -> index
      %41 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %43 = "emitc.load"(%42) : (!emitc.lvalue<f32>) -> f32
      %44 = "emitc.call_opaque"(%41, %43, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %46 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %47 = "emitc.load"(%46) : (!emitc.lvalue<f32>) -> f32
      %48 = "emitc.call_opaque"(%45, %47, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %37, %34) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%29, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %53 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %54 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %55 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %56 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %57 = "emitc.apply"(%56) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%57, %53, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %58 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %59 = "emitc.apply"(%58) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%59, %54, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %60 = "emitc.mul"(%12, %8) : (index, index) -> index
    %61 = "emitc.subscript"(%7, %60) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %62 = "emitc.apply"(%61) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%62, %55, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x4_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%23, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%27 : index):
      %28 = "emitc.mul"(%27, %3) : (index, index) -> index
      %29 = "emitc.constant"() {value = 4 : index} : () -> index
      %30 = "emitc.subscript"(%2, %28) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %32 = "emitc.call_opaque"(%31, %29) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %33 = "emitc.mul"(%27, %5) : (index, index) -> index
      %34 = "emitc.add"(%33, %11) : (index, index) -> index
      %35 = "emitc.add"(%33, %12) : (index, index) -> index
      %36 = "emitc.add"(%33, %13) : (index, index) -> index
      %37 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %38 = "emitc.subscript"(%4, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %39 = "emitc.load"(%38) : (!emitc.lvalue<f32>) -> f32
      %40 = "emitc.call_opaque"(%37, %39, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %42 = "emitc.subscript"(%4, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %43 = "emitc.load"(%42) : (!emitc.lvalue<f32>) -> f32
      %44 = "emitc.call_opaque"(%41, %43, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %46 = "emitc.subscript"(%4, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %47 = "emitc.load"(%46) : (!emitc.lvalue<f32>) -> f32
      %48 = "emitc.call_opaque"(%45, %47, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %32, %29) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%23, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%24, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%25, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %53 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %54 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %55 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %56 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %57 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %58 = "emitc.apply"(%57) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%58, %53, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %59 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %60 = "emitc.apply"(%59) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%60, %54, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %61 = "emitc.mul"(%12, %8) : (index, index) -> index
    %62 = "emitc.subscript"(%7, %61) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %63 = "emitc.apply"(%62) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%63, %55, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %64 = "emitc.mul"(%13, %8) : (index, index) -> index
    %65 = "emitc.subscript"(%7, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %66 = "emitc.apply"(%65) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%66, %56, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x4_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%33, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %34 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%34, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %35 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%35, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %36 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%36, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%37 : index):
      %38 = "emitc.mul"(%37, %3) : (index, index) -> index
      %39 = "emitc.constant"() {value = 4 : index} : () -> index
      %40 = "emitc.subscript"(%2, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %41 = "emitc.apply"(%40) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %42 = "emitc.call_opaque"(%41, %39) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %43 = "emitc.mul"(%37, %5) : (index, index) -> index
      %44 = "emitc.add"(%43, %11) : (index, index) -> index
      %45 = "emitc.add"(%43, %12) : (index, index) -> index
      %46 = "emitc.add"(%43, %13) : (index, index) -> index
      %47 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %49 = "emitc.load"(%48) : (!emitc.lvalue<f32>) -> f32
      %50 = "emitc.call_opaque"(%47, %49, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %53 = "emitc.load"(%52) : (!emitc.lvalue<f32>) -> f32
      %54 = "emitc.call_opaque"(%51, %53, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.load"(%35) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %57 = "emitc.load"(%56) : (!emitc.lvalue<f32>) -> f32
      %58 = "emitc.call_opaque"(%55, %57, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.load"(%36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %46) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %42, %39) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%33, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%34, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%35, %58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%36, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %63 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %64 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.load"(%35) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %66 = "emitc.load"(%36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %67 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %68 = "emitc.apply"(%67) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%68, %63, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %69 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %70 = "emitc.apply"(%69) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%70, %64, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %71 = "emitc.mul"(%12, %8) : (index, index) -> index
    %72 = "emitc.subscript"(%7, %71) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %73 = "emitc.apply"(%72) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%73, %65, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %74 = "emitc.mul"(%13, %8) : (index, index) -> index
    %75 = "emitc.subscript"(%7, %74) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %76 = "emitc.apply"(%75) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%76, %66, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x5_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%24, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%29 : index):
      %30 = "emitc.mul"(%29, %3) : (index, index) -> index
      %31 = "emitc.constant"() {value = 4 : index} : () -> index
      %32 = "emitc.subscript"(%2, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %33 = "emitc.apply"(%32) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %34 = "emitc.call_opaque"(%33, %31) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %35 = "emitc.mul"(%29, %5) : (index, index) -> index
      %36 = "emitc.add"(%35, %11) : (index, index) -> index
      %37 = "emitc.add"(%35, %12) : (index, index) -> index
      %38 = "emitc.add"(%35, %13) : (index, index) -> index
      %39 = "emitc.add"(%35, %14) : (index, index) -> index
      %40 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.subscript"(%4, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %42 = "emitc.load"(%41) : (!emitc.lvalue<f32>) -> f32
      %43 = "emitc.call_opaque"(%40, %42, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %44 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %45 = "emitc.subscript"(%4, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %46 = "emitc.load"(%45) : (!emitc.lvalue<f32>) -> f32
      %47 = "emitc.call_opaque"(%44, %46, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %49 = "emitc.subscript"(%4, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %50 = "emitc.load"(%49) : (!emitc.lvalue<f32>) -> f32
      %51 = "emitc.call_opaque"(%48, %50, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %54 = "emitc.load"(%53) : (!emitc.lvalue<f32>) -> f32
      %55 = "emitc.call_opaque"(%52, %54, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %58 = "emitc.load"(%57) : (!emitc.lvalue<f32>) -> f32
      %59 = "emitc.call_opaque"(%56, %58, %34, %31) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%24, %43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%25, %47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %59) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %60 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %61 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %62 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %63 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %64 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %66 = "emitc.apply"(%65) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%66, %60, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %67 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %68 = "emitc.apply"(%67) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%68, %61, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %69 = "emitc.mul"(%12, %8) : (index, index) -> index
    %70 = "emitc.subscript"(%7, %69) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %71 = "emitc.apply"(%70) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%71, %62, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %72 = "emitc.mul"(%13, %8) : (index, index) -> index
    %73 = "emitc.subscript"(%7, %72) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %74 = "emitc.apply"(%73) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%74, %63, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %75 = "emitc.mul"(%14, %8) : (index, index) -> index
    %76 = "emitc.subscript"(%7, %75) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %77 = "emitc.apply"(%76) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%77, %64, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x5_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%37, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %38 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%38, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %39 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%39, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %40 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%40, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %41 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%41, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%42 : index):
      %43 = "emitc.mul"(%42, %3) : (index, index) -> index
      %44 = "emitc.constant"() {value = 4 : index} : () -> index
      %45 = "emitc.subscript"(%2, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %46 = "emitc.apply"(%45) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %47 = "emitc.call_opaque"(%46, %44) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.mul"(%42, %5) : (index, index) -> index
      %49 = "emitc.add"(%48, %11) : (index, index) -> index
      %50 = "emitc.add"(%48, %12) : (index, index) -> index
      %51 = "emitc.add"(%48, %13) : (index, index) -> index
      %52 = "emitc.add"(%48, %14) : (index, index) -> index
      %53 = "emitc.load"(%37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.subscript"(%4, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %55 = "emitc.load"(%54) : (!emitc.lvalue<f32>) -> f32
      %56 = "emitc.call_opaque"(%53, %55, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.load"(%38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.subscript"(%4, %49) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %59 = "emitc.load"(%58) : (!emitc.lvalue<f32>) -> f32
      %60 = "emitc.call_opaque"(%57, %59, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %61 = "emitc.load"(%39) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.subscript"(%4, %50) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %63 = "emitc.load"(%62) : (!emitc.lvalue<f32>) -> f32
      %64 = "emitc.call_opaque"(%61, %63, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %65 = "emitc.load"(%40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %51) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %52) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %47, %44) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%37, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%38, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%39, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%40, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%41, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %73 = "emitc.load"(%37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %74 = "emitc.load"(%38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %75 = "emitc.load"(%39) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %76 = "emitc.load"(%40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %77 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %78 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %79 = "emitc.apply"(%78) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%79, %73, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %80 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %81 = "emitc.apply"(%80) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%81, %74, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %82 = "emitc.mul"(%12, %8) : (index, index) -> index
    %83 = "emitc.subscript"(%7, %82) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %84 = "emitc.apply"(%83) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%84, %75, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %85 = "emitc.mul"(%13, %8) : (index, index) -> index
    %86 = "emitc.subscript"(%7, %85) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %87 = "emitc.apply"(%86) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%87, %76, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %88 = "emitc.mul"(%14, %8) : (index, index) -> index
    %89 = "emitc.subscript"(%7, %88) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %77, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x6_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%25, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%31 : index):
      %32 = "emitc.mul"(%31, %3) : (index, index) -> index
      %33 = "emitc.constant"() {value = 4 : index} : () -> index
      %34 = "emitc.subscript"(%2, %32) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %36 = "emitc.call_opaque"(%35, %33) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %37 = "emitc.mul"(%31, %5) : (index, index) -> index
      %38 = "emitc.add"(%37, %11) : (index, index) -> index
      %39 = "emitc.add"(%37, %12) : (index, index) -> index
      %40 = "emitc.add"(%37, %13) : (index, index) -> index
      %41 = "emitc.add"(%37, %14) : (index, index) -> index
      %42 = "emitc.add"(%37, %15) : (index, index) -> index
      %43 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %44 = "emitc.subscript"(%4, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %45 = "emitc.load"(%44) : (!emitc.lvalue<f32>) -> f32
      %46 = "emitc.call_opaque"(%43, %45, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %47 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %48 = "emitc.subscript"(%4, %38) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %49 = "emitc.load"(%48) : (!emitc.lvalue<f32>) -> f32
      %50 = "emitc.call_opaque"(%47, %49, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %52 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %53 = "emitc.load"(%52) : (!emitc.lvalue<f32>) -> f32
      %54 = "emitc.call_opaque"(%51, %53, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %56 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %57 = "emitc.load"(%56) : (!emitc.lvalue<f32>) -> f32
      %58 = "emitc.call_opaque"(%55, %57, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %64 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %65 = "emitc.load"(%64) : (!emitc.lvalue<f32>) -> f32
      %66 = "emitc.call_opaque"(%63, %65, %36, %33) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%25, %46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%26, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %67 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %68 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %69 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %70 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %71 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %72 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %73 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %74 = "emitc.apply"(%73) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%74, %67, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %75 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %76 = "emitc.apply"(%75) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%76, %68, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %77 = "emitc.mul"(%12, %8) : (index, index) -> index
    %78 = "emitc.subscript"(%7, %77) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %79 = "emitc.apply"(%78) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%79, %69, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %80 = "emitc.mul"(%13, %8) : (index, index) -> index
    %81 = "emitc.subscript"(%7, %80) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %82 = "emitc.apply"(%81) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%82, %70, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %83 = "emitc.mul"(%14, %8) : (index, index) -> index
    %84 = "emitc.subscript"(%7, %83) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %85 = "emitc.apply"(%84) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%85, %71, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %86 = "emitc.mul"(%15, %8) : (index, index) -> index
    %87 = "emitc.subscript"(%7, %86) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %88 = "emitc.apply"(%87) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%88, %72, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x6_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%41, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %42 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%42, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %43 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%43, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %44 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%44, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %45 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%45, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %46 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%46, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%47 : index):
      %48 = "emitc.mul"(%47, %3) : (index, index) -> index
      %49 = "emitc.constant"() {value = 4 : index} : () -> index
      %50 = "emitc.subscript"(%2, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.apply"(%50) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %52 = "emitc.call_opaque"(%51, %49) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.mul"(%47, %5) : (index, index) -> index
      %54 = "emitc.add"(%53, %11) : (index, index) -> index
      %55 = "emitc.add"(%53, %12) : (index, index) -> index
      %56 = "emitc.add"(%53, %13) : (index, index) -> index
      %57 = "emitc.add"(%53, %14) : (index, index) -> index
      %58 = "emitc.add"(%53, %15) : (index, index) -> index
      %59 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %60 = "emitc.subscript"(%4, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<f32>) -> f32
      %62 = "emitc.call_opaque"(%59, %61, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.load"(%42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %64 = "emitc.subscript"(%4, %54) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %65 = "emitc.load"(%64) : (!emitc.lvalue<f32>) -> f32
      %66 = "emitc.call_opaque"(%63, %65, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %67 = "emitc.load"(%43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %68 = "emitc.subscript"(%4, %55) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %69 = "emitc.load"(%68) : (!emitc.lvalue<f32>) -> f32
      %70 = "emitc.call_opaque"(%67, %69, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %71 = "emitc.load"(%44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %72 = "emitc.subscript"(%4, %56) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %73 = "emitc.load"(%72) : (!emitc.lvalue<f32>) -> f32
      %74 = "emitc.call_opaque"(%71, %73, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %75 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %76 = "emitc.subscript"(%4, %57) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %77 = "emitc.load"(%76) : (!emitc.lvalue<f32>) -> f32
      %78 = "emitc.call_opaque"(%75, %77, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %79 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %80 = "emitc.subscript"(%4, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %81 = "emitc.load"(%80) : (!emitc.lvalue<f32>) -> f32
      %82 = "emitc.call_opaque"(%79, %81, %52, %49) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%41, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%42, %66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%43, %70) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%44, %74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%45, %78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%46, %82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %83 = "emitc.load"(%41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %84 = "emitc.load"(%42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %85 = "emitc.load"(%43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %86 = "emitc.load"(%44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %87 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %88 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %89 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %83, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %92 = "emitc.apply"(%91) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%92, %84, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %93 = "emitc.mul"(%12, %8) : (index, index) -> index
    %94 = "emitc.subscript"(%7, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%95, %85, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %96 = "emitc.mul"(%13, %8) : (index, index) -> index
    %97 = "emitc.subscript"(%7, %96) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %98 = "emitc.apply"(%97) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%98, %86, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %99 = "emitc.mul"(%14, %8) : (index, index) -> index
    %100 = "emitc.subscript"(%7, %99) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %87, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.mul"(%15, %8) : (index, index) -> index
    %103 = "emitc.subscript"(%7, %102) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %104 = "emitc.apply"(%103) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%104, %88, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x7_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%26, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %32 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%32, %25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%33 : index):
      %34 = "emitc.mul"(%33, %3) : (index, index) -> index
      %35 = "emitc.constant"() {value = 4 : index} : () -> index
      %36 = "emitc.subscript"(%2, %34) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %38 = "emitc.call_opaque"(%37, %35) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %39 = "emitc.mul"(%33, %5) : (index, index) -> index
      %40 = "emitc.add"(%39, %11) : (index, index) -> index
      %41 = "emitc.add"(%39, %12) : (index, index) -> index
      %42 = "emitc.add"(%39, %13) : (index, index) -> index
      %43 = "emitc.add"(%39, %14) : (index, index) -> index
      %44 = "emitc.add"(%39, %15) : (index, index) -> index
      %45 = "emitc.add"(%39, %16) : (index, index) -> index
      %46 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %47 = "emitc.subscript"(%4, %39) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %48 = "emitc.load"(%47) : (!emitc.lvalue<f32>) -> f32
      %49 = "emitc.call_opaque"(%46, %48, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %51 = "emitc.subscript"(%4, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %52 = "emitc.load"(%51) : (!emitc.lvalue<f32>) -> f32
      %53 = "emitc.call_opaque"(%50, %52, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %55 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %56 = "emitc.load"(%55) : (!emitc.lvalue<f32>) -> f32
      %57 = "emitc.call_opaque"(%54, %56, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %59 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %60 = "emitc.load"(%59) : (!emitc.lvalue<f32>) -> f32
      %61 = "emitc.call_opaque"(%58, %60, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %64 = "emitc.load"(%63) : (!emitc.lvalue<f32>) -> f32
      %65 = "emitc.call_opaque"(%62, %64, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %67 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %68 = "emitc.load"(%67) : (!emitc.lvalue<f32>) -> f32
      %69 = "emitc.call_opaque"(%66, %68, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %71 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %72 = "emitc.load"(%71) : (!emitc.lvalue<f32>) -> f32
      %73 = "emitc.call_opaque"(%70, %72, %38, %35) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%26, %49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%27, %53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %57) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %61) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %65) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %69) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%32, %73) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %74 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %75 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %76 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %77 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %78 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %79 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %80 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %81 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %82 = "emitc.apply"(%81) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%82, %74, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %83 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %84 = "emitc.apply"(%83) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%84, %75, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %85 = "emitc.mul"(%12, %8) : (index, index) -> index
    %86 = "emitc.subscript"(%7, %85) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %87 = "emitc.apply"(%86) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%87, %76, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %88 = "emitc.mul"(%13, %8) : (index, index) -> index
    %89 = "emitc.subscript"(%7, %88) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %77, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.mul"(%14, %8) : (index, index) -> index
    %92 = "emitc.subscript"(%7, %91) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %93 = "emitc.apply"(%92) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%93, %78, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %94 = "emitc.mul"(%15, %8) : (index, index) -> index
    %95 = "emitc.subscript"(%7, %94) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %96 = "emitc.apply"(%95) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%96, %79, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %97 = "emitc.mul"(%16, %8) : (index, index) -> index
    %98 = "emitc.subscript"(%7, %97) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %99 = "emitc.apply"(%98) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%99, %80, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x7_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.mul"(%16, %8) : (index, index) -> index
    %42 = "emitc.subscript"(%7, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %43 = "emitc.apply"(%42) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %44 = "emitc.call_opaque"(%43, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%45, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %46 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%46, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %47 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%47, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %48 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%48, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %49 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%49, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %50 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%50, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %51 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%51, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%52 : index):
      %53 = "emitc.mul"(%52, %3) : (index, index) -> index
      %54 = "emitc.constant"() {value = 4 : index} : () -> index
      %55 = "emitc.subscript"(%2, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %56 = "emitc.apply"(%55) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %57 = "emitc.call_opaque"(%56, %54) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.mul"(%52, %5) : (index, index) -> index
      %59 = "emitc.add"(%58, %11) : (index, index) -> index
      %60 = "emitc.add"(%58, %12) : (index, index) -> index
      %61 = "emitc.add"(%58, %13) : (index, index) -> index
      %62 = "emitc.add"(%58, %14) : (index, index) -> index
      %63 = "emitc.add"(%58, %15) : (index, index) -> index
      %64 = "emitc.add"(%58, %16) : (index, index) -> index
      %65 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %59) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %73 = "emitc.load"(%47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %74 = "emitc.subscript"(%4, %60) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %75 = "emitc.load"(%74) : (!emitc.lvalue<f32>) -> f32
      %76 = "emitc.call_opaque"(%73, %75, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %77 = "emitc.load"(%48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %78 = "emitc.subscript"(%4, %61) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %79 = "emitc.load"(%78) : (!emitc.lvalue<f32>) -> f32
      %80 = "emitc.call_opaque"(%77, %79, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %81 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %82 = "emitc.subscript"(%4, %62) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %83 = "emitc.load"(%82) : (!emitc.lvalue<f32>) -> f32
      %84 = "emitc.call_opaque"(%81, %83, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %85 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %86 = "emitc.subscript"(%4, %63) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %87 = "emitc.load"(%86) : (!emitc.lvalue<f32>) -> f32
      %88 = "emitc.call_opaque"(%85, %87, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %89 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %90 = "emitc.subscript"(%4, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %91 = "emitc.load"(%90) : (!emitc.lvalue<f32>) -> f32
      %92 = "emitc.call_opaque"(%89, %91, %57, %54) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%45, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%46, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%47, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%48, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%49, %84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%50, %88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%51, %92) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %93 = "emitc.load"(%45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %94 = "emitc.load"(%46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %95 = "emitc.load"(%47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %96 = "emitc.load"(%48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %97 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %98 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %99 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %100 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %93, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %103 = "emitc.apply"(%102) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%103, %94, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %104 = "emitc.mul"(%12, %8) : (index, index) -> index
    %105 = "emitc.subscript"(%7, %104) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %106 = "emitc.apply"(%105) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%106, %95, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %107 = "emitc.mul"(%13, %8) : (index, index) -> index
    %108 = "emitc.subscript"(%7, %107) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %109 = "emitc.apply"(%108) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%109, %96, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %110 = "emitc.mul"(%14, %8) : (index, index) -> index
    %111 = "emitc.subscript"(%7, %110) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %112 = "emitc.apply"(%111) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%112, %97, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %113 = "emitc.mul"(%15, %8) : (index, index) -> index
    %114 = "emitc.subscript"(%7, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %115 = "emitc.apply"(%114) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%115, %98, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %116 = "emitc.mul"(%16, %8) : (index, index) -> index
    %117 = "emitc.subscript"(%7, %116) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %118 = "emitc.apply"(%117) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%118, %99, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x8_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %20 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %21 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %23 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %24 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %26 = "emitc.call_opaque"(%9, %14) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %27 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%27, %19) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %28 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%28, %20) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %29 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%29, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %30 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%30, %22) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%31, %23) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %32 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%32, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %33 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%33, %25) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %34 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%34, %26) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%35 : index):
      %36 = "emitc.mul"(%35, %3) : (index, index) -> index
      %37 = "emitc.constant"() {value = 4 : index} : () -> index
      %38 = "emitc.subscript"(%2, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %40 = "emitc.call_opaque"(%39, %37) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %41 = "emitc.mul"(%35, %5) : (index, index) -> index
      %42 = "emitc.add"(%41, %11) : (index, index) -> index
      %43 = "emitc.add"(%41, %12) : (index, index) -> index
      %44 = "emitc.add"(%41, %13) : (index, index) -> index
      %45 = "emitc.add"(%41, %14) : (index, index) -> index
      %46 = "emitc.add"(%41, %15) : (index, index) -> index
      %47 = "emitc.add"(%41, %16) : (index, index) -> index
      %48 = "emitc.add"(%41, %17) : (index, index) -> index
      %49 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %50 = "emitc.subscript"(%4, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<f32>) -> f32
      %52 = "emitc.call_opaque"(%49, %51, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %53 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %54 = "emitc.subscript"(%4, %42) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %55 = "emitc.load"(%54) : (!emitc.lvalue<f32>) -> f32
      %56 = "emitc.call_opaque"(%53, %55, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %57 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %58 = "emitc.subscript"(%4, %43) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %59 = "emitc.load"(%58) : (!emitc.lvalue<f32>) -> f32
      %60 = "emitc.call_opaque"(%57, %59, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %61 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %62 = "emitc.subscript"(%4, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %63 = "emitc.load"(%62) : (!emitc.lvalue<f32>) -> f32
      %64 = "emitc.call_opaque"(%61, %63, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %65 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %66 = "emitc.subscript"(%4, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<f32>) -> f32
      %68 = "emitc.call_opaque"(%65, %67, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %69 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %70 = "emitc.subscript"(%4, %46) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %71 = "emitc.load"(%70) : (!emitc.lvalue<f32>) -> f32
      %72 = "emitc.call_opaque"(%69, %71, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %73 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %74 = "emitc.subscript"(%4, %47) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %75 = "emitc.load"(%74) : (!emitc.lvalue<f32>) -> f32
      %76 = "emitc.call_opaque"(%73, %75, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %77 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %78 = "emitc.subscript"(%4, %48) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %79 = "emitc.load"(%78) : (!emitc.lvalue<f32>) -> f32
      %80 = "emitc.call_opaque"(%77, %79, %40, %37) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%27, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%28, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%29, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%30, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%31, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%32, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%33, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%34, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %81 = "emitc.load"(%27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %82 = "emitc.load"(%28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %83 = "emitc.load"(%29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %84 = "emitc.load"(%30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %85 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %86 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %87 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %88 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %89 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %90 = "emitc.apply"(%89) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%90, %81, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %91 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %92 = "emitc.apply"(%91) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%92, %82, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %93 = "emitc.mul"(%12, %8) : (index, index) -> index
    %94 = "emitc.subscript"(%7, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%95, %83, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %96 = "emitc.mul"(%13, %8) : (index, index) -> index
    %97 = "emitc.subscript"(%7, %96) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %98 = "emitc.apply"(%97) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%98, %84, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %99 = "emitc.mul"(%14, %8) : (index, index) -> index
    %100 = "emitc.subscript"(%7, %99) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%101, %85, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %102 = "emitc.mul"(%15, %8) : (index, index) -> index
    %103 = "emitc.subscript"(%7, %102) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %104 = "emitc.apply"(%103) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%104, %86, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %105 = "emitc.mul"(%16, %8) : (index, index) -> index
    %106 = "emitc.subscript"(%7, %105) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %107 = "emitc.apply"(%106) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%107, %87, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %108 = "emitc.mul"(%17, %8) : (index, index) -> index
    %109 = "emitc.subscript"(%7, %108) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %110 = "emitc.apply"(%109) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%110, %88, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR8_4x8_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<f32>, %2 : !emitc.ptr<f32>, %3 : index, %4 : !emitc.ptr<f32>, %5 : index, %6 : !emitc.ptr<f32>, %7 : !emitc.ptr<f32>, %8 : index):
    %9 = "emitc.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 2 : index} : () -> index
    %13 = "emitc.constant"() {value = 3 : index} : () -> index
    %14 = "emitc.constant"() {value = 4 : index} : () -> index
    %15 = "emitc.constant"() {value = 5 : index} : () -> index
    %16 = "emitc.constant"() {value = 6 : index} : () -> index
    %17 = "emitc.constant"() {value = 7 : index} : () -> index
    %18 = "emitc.constant"() {value = 8 : index} : () -> index
    %19 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %20 = "emitc.apply"(%19) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %21 = "emitc.call_opaque"(%20, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %22 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %24 = "emitc.call_opaque"(%23, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %25 = "emitc.mul"(%12, %8) : (index, index) -> index
    %26 = "emitc.subscript"(%7, %25) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %27 = "emitc.apply"(%26) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %28 = "emitc.call_opaque"(%27, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.mul"(%13, %8) : (index, index) -> index
    %30 = "emitc.subscript"(%7, %29) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%31, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.mul"(%14, %8) : (index, index) -> index
    %34 = "emitc.subscript"(%7, %33) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.mul"(%15, %8) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.mul"(%16, %8) : (index, index) -> index
    %42 = "emitc.subscript"(%7, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %43 = "emitc.apply"(%42) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %44 = "emitc.call_opaque"(%43, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.mul"(%17, %8) : (index, index) -> index
    %46 = "emitc.subscript"(%7, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %47 = "emitc.apply"(%46) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %48 = "emitc.call_opaque"(%47, %14) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %49 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%49, %21) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %50 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%50, %24) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %51 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%51, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %52 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%52, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %53 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%53, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %54 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%54, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %55 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%55, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %56 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%56, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%57 : index):
      %58 = "emitc.mul"(%57, %3) : (index, index) -> index
      %59 = "emitc.constant"() {value = 4 : index} : () -> index
      %60 = "emitc.subscript"(%2, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %61 = "emitc.apply"(%60) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %62 = "emitc.call_opaque"(%61, %59) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %63 = "emitc.mul"(%57, %5) : (index, index) -> index
      %64 = "emitc.add"(%63, %11) : (index, index) -> index
      %65 = "emitc.add"(%63, %12) : (index, index) -> index
      %66 = "emitc.add"(%63, %13) : (index, index) -> index
      %67 = "emitc.add"(%63, %14) : (index, index) -> index
      %68 = "emitc.add"(%63, %15) : (index, index) -> index
      %69 = "emitc.add"(%63, %16) : (index, index) -> index
      %70 = "emitc.add"(%63, %17) : (index, index) -> index
      %71 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %72 = "emitc.subscript"(%4, %63) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %73 = "emitc.load"(%72) : (!emitc.lvalue<f32>) -> f32
      %74 = "emitc.call_opaque"(%71, %73, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %75 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %76 = "emitc.subscript"(%4, %64) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %77 = "emitc.load"(%76) : (!emitc.lvalue<f32>) -> f32
      %78 = "emitc.call_opaque"(%75, %77, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %79 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %80 = "emitc.subscript"(%4, %65) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %81 = "emitc.load"(%80) : (!emitc.lvalue<f32>) -> f32
      %82 = "emitc.call_opaque"(%79, %81, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %83 = "emitc.load"(%52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %84 = "emitc.subscript"(%4, %66) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %85 = "emitc.load"(%84) : (!emitc.lvalue<f32>) -> f32
      %86 = "emitc.call_opaque"(%83, %85, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %87 = "emitc.load"(%53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %88 = "emitc.subscript"(%4, %67) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %89 = "emitc.load"(%88) : (!emitc.lvalue<f32>) -> f32
      %90 = "emitc.call_opaque"(%87, %89, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %91 = "emitc.load"(%54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %92 = "emitc.subscript"(%4, %68) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %93 = "emitc.load"(%92) : (!emitc.lvalue<f32>) -> f32
      %94 = "emitc.call_opaque"(%91, %93, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %95 = "emitc.load"(%55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %96 = "emitc.subscript"(%4, %69) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %97 = "emitc.load"(%96) : (!emitc.lvalue<f32>) -> f32
      %98 = "emitc.call_opaque"(%95, %97, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %99 = "emitc.load"(%56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %100 = "emitc.subscript"(%4, %70) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %101 = "emitc.load"(%100) : (!emitc.lvalue<f32>) -> f32
      %102 = "emitc.call_opaque"(%99, %101, %62, %59) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%49, %74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%50, %78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%51, %82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%52, %86) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%53, %90) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%54, %94) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%55, %98) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%56, %102) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %103 = "emitc.load"(%49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %104 = "emitc.load"(%50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %105 = "emitc.load"(%51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %106 = "emitc.load"(%52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %107 = "emitc.load"(%53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %108 = "emitc.load"(%54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %109 = "emitc.load"(%55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %110 = "emitc.load"(%56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %111 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %112 = "emitc.apply"(%111) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%112, %103, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %113 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %114 = "emitc.apply"(%113) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%114, %104, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %115 = "emitc.mul"(%12, %8) : (index, index) -> index
    %116 = "emitc.subscript"(%7, %115) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %117 = "emitc.apply"(%116) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%117, %105, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %118 = "emitc.mul"(%13, %8) : (index, index) -> index
    %119 = "emitc.subscript"(%7, %118) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %120 = "emitc.apply"(%119) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%120, %106, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %121 = "emitc.mul"(%14, %8) : (index, index) -> index
    %122 = "emitc.subscript"(%7, %121) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %123 = "emitc.apply"(%122) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%123, %107, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %124 = "emitc.mul"(%15, %8) : (index, index) -> index
    %125 = "emitc.subscript"(%7, %124) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %126 = "emitc.apply"(%125) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%126, %108, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %127 = "emitc.mul"(%16, %8) : (index, index) -> index
    %128 = "emitc.subscript"(%7, %127) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %129 = "emitc.apply"(%128) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%129, %109, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %130 = "emitc.mul"(%17, %8) : (index, index) -> index
    %131 = "emitc.subscript"(%7, %130) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %132 = "emitc.apply"(%131) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%132, %110, %14) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
  "func.func"() <{sym_name = "gemm_macro_v_fp32_MR4_NR8", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>):
    %13 = "emitc.constant"() {value = 0 : i32} : () -> i32
    %14 = "emitc.constant"() {value = 1365 : i32} : () -> i32
    "emitc.for"(%13, %1, %14) ({
    ^bb1(%15 : i32):
      %16 = "emitc.sub"(%1, %15) : (i32, i32) -> i32
      %17 = "emitc.constant"() {value = 1365 : i32} : () -> i32
      %18 = "emitc.call_opaque"(%16, %17) <{callee = "std::min"}> : (i32, i32) -> i32
      %19 = "emitc.constant"() {value = 512 : i32} : () -> i32
      "emitc.for"(%13, %2, %19) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.constant"() {value = 512 : i32} : () -> i32
        %23 = "emitc.call_opaque"(%21, %22) <{callee = "std::min"}> : (i32, i32) -> i32
        %24 = "emitc.mul"(%15, %7) : (i32, i32) -> i32
        %25 = "emitc.add"(%24, %20) : (i32, i32) -> i32
        %26 = "emitc.call_opaque"(%6, %25) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%26, %12, %23, %18, %7) <{callee = "pack_CB_v_if_fp32_MR4_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %27 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        %28 = "emitc.constant"() {value = 192 : i32} : () -> i32
        "emitc.for"(%13, %0, %28) ({
        ^bb3(%29 : i32):
          %30 = "emitc.sub"(%0, %29) : (i32, i32) -> i32
          %31 = "emitc.constant"() {value = 192 : i32} : () -> i32
          %32 = "emitc.call_opaque"(%30, %31) <{callee = "std::min"}> : (i32, i32) -> i32
          %33 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %34 = "emitc.add"(%33, %29) : (i32, i32) -> i32
          %35 = "emitc.call_opaque"(%4, %34) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%35, %11, %32, %23, %5) <{callee = "pack_RB_v_if_fp32_MR4_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %36 = "emitc.constant"() {value = 8 : i32} : () -> i32
          "emitc.for"(%13, %18, %36) ({
          ^bb4(%37 : i32):
            %38 = "emitc.sub"(%18, %37) : (i32, i32) -> i32
            %39 = "emitc.constant"() {value = 8 : i32} : () -> i32
            %40 = "emitc.call_opaque"(%38, %39) <{callee = "std::min"}> : (i32, i32) -> i32
            %41 = "emitc.constant"() {value = 4 : i32} : () -> i32
            "emitc.for"(%13, %32, %41) ({
            ^bb5(%42 : i32):
              %43 = "emitc.sub"(%32, %42) : (i32, i32) -> i32
              %44 = "emitc.constant"() {value = 4 : i32} : () -> i32
              %45 = "emitc.call_opaque"(%43, %44) <{callee = "std::min"}> : (i32, i32) -> i32
              %46 = "emitc.add"(%29, %42) : (i32, i32) -> i32
              %47 = "emitc.add"(%15, %37) : (i32, i32) -> i32
              %48 = "emitc.mul"(%47, %10) : (i32, i32) -> i32
              %49 = "emitc.add"(%48, %46) : (i32, i32) -> i32
              %50 = "emitc.call_opaque"(%9, %49) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %51 = "emitc.mul"(%42, %23) : (i32, i32) -> i32
              %52 = "emitc.call_opaque"(%11, %51) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %53 = "emitc.mul"(%37, %23) : (i32, i32) -> i32
              %54 = "emitc.call_opaque"(%12, %53) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%45, %40, %23, %3, %52, %54, %27, %50, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR4_NR8"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
              "emitc.yield"() : () -> ()
            }) : (i32, i32, i32) -> ()
            "emitc.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "emitc.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "emitc.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "emitc.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>) -> i32} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_fp32_MR4_NR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 8 : index} : () -> index
    "emitc.for"(%5, %3, %8) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %8) <{callee = "std::min"}> : (index, index) -> index
      %12 = "emitc.call_opaque"(%11) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
      %13 = "emitc.mul"(%4, %7) : (index, index) -> index
      "emitc.for"(%5, %2, %6) ({
      ^bb2(%14 : index):
        %15 = "emitc.mul"(%9, %4) : (index, index) -> index
        %16 = "emitc.add"(%15, %14) : (index, index) -> index
        %17 = "emitc.subscript"(%0, %16) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %18 = "emitc.apply"(%17) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %19 = "emitc.call_opaque"(%18, %13, %12) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %20 = "emitc.mul"(%9, %2) : (index, index) -> index
        %21 = "emitc.mul"(%14, %8) : (index, index) -> index
        %22 = "emitc.add"(%20, %21) : (index, index) -> index
        %23 = "emitc.subscript"(%1, %22) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %24 = "emitc.apply"(%23) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%24, %19, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_fp32_MR4_NR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 8 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%8 : index):
      %9 = "emitc.sub"(%3, %8) : (index, index) -> index
      %10 = "emitc.call_opaque"(%9, %7) <{callee = "std::min"}> : (index, index) -> index
      %11 = "emitc.call_opaque"(%10) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
      "emitc.for"(%5, %2, %6) ({
      ^bb2(%12 : index):
        %13 = "emitc.mul"(%12, %4) : (index, index) -> index
        %14 = "emitc.add"(%13, %8) : (index, index) -> index
        %15 = "emitc.subscript"(%0, %14) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %16 = "emitc.apply"(%15) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %17 = "emitc.call_opaque"(%16, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %18 = "emitc.mul"(%8, %2) : (index, index) -> index
        %19 = "emitc.mul"(%12, %7) : (index, index) -> index
        %20 = "emitc.add"(%18, %19) : (index, index) -> index
        %21 = "emitc.subscript"(%1, %20) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %22 = "emitc.apply"(%21) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%22, %17, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_fp32_MR4_NR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%8 : index):
      %9 = "emitc.sub"(%2, %8) : (index, index) -> index
      %10 = "emitc.call_opaque"(%9, %7) <{callee = "std::min"}> : (index, index) -> index
      %11 = "emitc.call_opaque"(%10) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
      "emitc.for"(%5, %3, %6) ({
      ^bb2(%12 : index):
        %13 = "emitc.mul"(%12, %4) : (index, index) -> index
        %14 = "emitc.add"(%13, %8) : (index, index) -> index
        %15 = "emitc.subscript"(%0, %14) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %16 = "emitc.apply"(%15) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %17 = "emitc.call_opaque"(%16, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %18 = "emitc.mul"(%8, %3) : (index, index) -> index
        %19 = "emitc.mul"(%12, %7) : (index, index) -> index
        %20 = "emitc.add"(%18, %19) : (index, index) -> index
        %21 = "emitc.subscript"(%1, %20) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %22 = "emitc.apply"(%21) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%22, %17, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_fp32_MR4_NR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %8) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%2, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %8) <{callee = "std::min"}> : (index, index) -> index
      %12 = "emitc.call_opaque"(%11) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
      %13 = "emitc.mul"(%4, %7) : (index, index) -> index
      "emitc.for"(%5, %3, %6) ({
      ^bb2(%14 : index):
        %15 = "emitc.mul"(%9, %4) : (index, index) -> index
        %16 = "emitc.add"(%15, %14) : (index, index) -> index
        %17 = "emitc.subscript"(%0, %16) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %18 = "emitc.apply"(%17) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %19 = "emitc.call_opaque"(%18, %13, %12) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %20 = "emitc.mul"(%9, %3) : (index, index) -> index
        %21 = "emitc.mul"(%14, %8) : (index, index) -> index
        %22 = "emitc.add"(%20, %21) : (index, index) -> index
        %23 = "emitc.subscript"(%1, %22) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %24 = "emitc.apply"(%23) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%24, %19, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
}) : () -> ()