"builtin.module"() ({
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR2_NR2_1x1_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR2_NR2_1x1_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR2_NR2_1x2_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR2_NR2_1x2_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR2_NR2_2x1_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR2_NR2_2x1_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR2_NR2_2x2_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR2_NR2_2x2_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_macro_v_fp32_MR2_NR2", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR2_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR2_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 2 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 2 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 2 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 2 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR2_NR2"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_fp32_MR2_NR2", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 2 : index} : () -> index
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
  "func.func"() <{sym_name = "pack_CB_v_else_fp32_MR2_NR2", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 2 : index} : () -> index
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
  "func.func"() <{sym_name = "pack_RB_v_if_fp32_MR2_NR2", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 2 : index} : () -> index
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
  "func.func"() <{sym_name = "pack_RB_v_else_fp32_MR2_NR2", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 2 : index} : () -> index
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