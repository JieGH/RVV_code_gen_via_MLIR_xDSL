"builtin.module"() ({
  "func.func"() <{sym_name = "pack_CB_v_if_fp32_RR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
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
  "func.func"() <{sym_name = "pack_CB_v_else_fp32_RR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
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
  "func.func"() <{sym_name = "pack_RB_v_if_fp32_RR4", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
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
  "func.func"() <{sym_name = "pack_RB_v_else_fp32_RR4", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
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
