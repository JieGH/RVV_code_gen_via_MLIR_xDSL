"builtin.module"() ({
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR1", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 1 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR1", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR1", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 1 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR1", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR2", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 2 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR2", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR2", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 2 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR2", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR3", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 3 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR3", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR3", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 3 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR3", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR4", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR4", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR4", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR4", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR5", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 5 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR5", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR5", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 5 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR5", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR6", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 6 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR6", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR6", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 6 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR6", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR7", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 7 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR7", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR7", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 7 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR7", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 8 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 8 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR9", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 9 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR9", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR9", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 9 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR9", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR10", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 10 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR10", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR10", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 10 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR10", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR11", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 11 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR11", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR11", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 11 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR11", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR12", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 12 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR12", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR12", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 12 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR12", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR13", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 13 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR13", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR13", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 13 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR13", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR14", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 14 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR14", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR14", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 14 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR14", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR15", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 15 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR15", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR15", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 15 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR15", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR16", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 16 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR16", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR16", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 16 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR16", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR17", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 17 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR17", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR17", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 17 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR17", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR18", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 18 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR18", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR18", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 18 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR18", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR19", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 19 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR19", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR19", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 19 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR19", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR20", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 20 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR20", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR20", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 20 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR20", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR21", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 21 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR21", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR21", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 21 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR21", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR22", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 22 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR22", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR22", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 22 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR22", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR23", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 23 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR23", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR23", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 23 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR23", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR24", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 24 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR24", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR24", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 24 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR24", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR25", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 25 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR25", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR25", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 25 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR25", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR26", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 26 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR26", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR26", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 26 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR26", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR27", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 27 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR27", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR27", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 27 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR27", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR28", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 28 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR28", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR28", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 28 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR28", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR29", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 29 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR29", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR29", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 29 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR29", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR30", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 30 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR30", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR30", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 30 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR30", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR31", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 31 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR31", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR31", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 31 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR31", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR32", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 32 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %8) ({
      ^bb2(%12 : index):
        %13 = "emitc.sub"(%2, %12) : (index, index) -> index
        %14 = "emitc.call_opaque"(%13, %8) <{callee = "std::min"}> : (index, index) -> index
        %15 = "emitc.call_opaque"(%14) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        "emitc.for"(%5, %11, %6) ({
        ^bb3(%16 : index):
          %17 = "emitc.add"(%9, %16) : (index, index) -> index
          %18 = "emitc.mul"(%17, %4) : (index, index) -> index
          %19 = "emitc.add"(%18, %12) : (index, index) -> index
          %20 = "emitc.subscript"(%0, %19) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %21 = "emitc.apply"(%20) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %22 = "emitc.call_opaque"(%21, %15) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
          %23 = "emitc.constant"() {value = 3 : index} : () -> index
          %24 = "emitc.add"(%2, %23) : (index, index) -> index
          %25 = "emitc.div"(%24, %8) : (index, index) -> index
          %26 = "emitc.mul"(%25, %8) : (index, index) -> index
          %27 = "emitc.div"(%9, %7) : (index, index) -> index
          %28 = "emitc.mul"(%26, %7) : (index, index) -> index
          %29 = "emitc.mul"(%27, %28) : (index, index) -> index
          %30 = "emitc.div"(%12, %8) : (index, index) -> index
          %31 = "emitc.mul"(%7, %8) : (index, index) -> index
          %32 = "emitc.mul"(%30, %31) : (index, index) -> index
          %33 = "emitc.mul"(%16, %8) : (index, index) -> index
          %34 = "emitc.add"(%29, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %22, %15) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_FP32_LANE_Pack_fp32_NR32", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_FP32_LANE_Pack_fp32_MR32", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 32 : index} : () -> index
    %8 = "emitc.constant"() {value = 4 : index} : () -> index
    %9 = "emitc.constant"() {value = 4 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
          %25 = "emitc.constant"() {value = 3 : index} : () -> index
          %26 = "emitc.add"(%3, %25) : (index, index) -> index
          %27 = "emitc.div"(%26, %8) : (index, index) -> index
          %28 = "emitc.mul"(%27, %8) : (index, index) -> index
          %29 = "emitc.div"(%10, %7) : (index, index) -> index
          %30 = "emitc.mul"(%28, %7) : (index, index) -> index
          %31 = "emitc.mul"(%29, %30) : (index, index) -> index
          %32 = "emitc.mul"(%13, %7) : (index, index) -> index
          %33 = "emitc.mul"(%18, %8) : (index, index) -> index
          %34 = "emitc.add"(%31, %32) : (index, index) -> index
          %35 = "emitc.add"(%34, %33) : (index, index) -> index
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_FP32_LANE_Pack_fp32_MR32", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
}) : () -> ()