"builtin.module"() ({
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_1x1_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_1x1_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_1x2_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_1x2_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_1x3_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_1x3_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_1x4_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_1x4_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_2x1_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_2x1_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_2x2_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_2x2_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_2x3_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_2x3_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_2x4_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_2x4_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_3x1_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_3x1_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_3x2_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_3x2_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_3x3_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_3x3_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_3x4_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_3x4_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_4x1_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_4x1_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_4x2_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_4x2_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_4x3_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_4x3_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_4x4_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_MR4_NR4_4x4_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
  "func.func"() <{sym_name = "gemm_macro_v_fp32_MR4_NR4", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>) -> ()}> ({
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
        "emitc.call_opaque"(%26, %12, %23, %18, %7) <{callee = "pack_CB_v_if_fp32_MR4_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
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
          "emitc.call_opaque"(%35, %11, %32, %23, %5) <{callee = "pack_RB_v_if_fp32_MR4_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %36 = "emitc.constant"() {value = 4 : i32} : () -> i32
          "emitc.for"(%13, %18, %36) ({
          ^bb4(%37 : i32):
            %38 = "emitc.sub"(%18, %37) : (i32, i32) -> i32
            %39 = "emitc.constant"() {value = 4 : i32} : () -> i32
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
              "emitc.call_opaque"(%45, %40, %23, %3, %52, %54, %27, %50, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR4_NR4"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  "func.func"() <{sym_name = "pack_CB_v_if_fp32_MR4_NR4", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
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
  "func.func"() <{sym_name = "pack_CB_v_else_fp32_MR4_NR4", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
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
  "func.func"() <{sym_name = "pack_RB_v_if_fp32_MR4_NR4", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
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
  "func.func"() <{sym_name = "pack_RB_v_else_fp32_MR4_NR4", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
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