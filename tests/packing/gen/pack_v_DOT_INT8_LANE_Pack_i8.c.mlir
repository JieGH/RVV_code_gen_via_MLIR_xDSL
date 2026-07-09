"builtin.module"() ({
  "func.func"() <{sym_name = "pack_CB_v_if_DOT_INT8_LANE_Pack_i8_RR8", function_type = (!emitc.ptr<i8>, !emitc.ptr<i8>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<i8>, %1 : !emitc.ptr<i8>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 8 : index} : () -> index
    %9 = "emitc.constant"() {value = 32 : index} : () -> index
    "emitc.for"(%5, %3, %8) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%3, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %8) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %9) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%2, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %9) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%12) <{callee = "__riscv_vsetvl_e8m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %6) : (index, index) -> index
        %18 = "emitc.div"(%15, %7) : (index, index) -> index
        %19 = "emitc.mul"(%18, %7) : (index, index) -> index
        "emitc.for"(%5, %19, %7) ({
        ^bb3(%20 : index):
          %21 = "emitc.mul"(%10, %4) : (index, index) -> index
          %22 = "emitc.add"(%13, %20) : (index, index) -> index
          %23 = "emitc.add"(%21, %22) : (index, index) -> index
          %24 = "emitc.constant"() {value = 1 : index} : () -> index
          %25 = "emitc.constant"() {value = 2 : index} : () -> index
          %26 = "emitc.constant"() {value = 3 : index} : () -> index
          %27 = "emitc.add"(%23, %24) : (index, index) -> index
          %28 = "emitc.add"(%23, %25) : (index, index) -> index
          %29 = "emitc.add"(%23, %26) : (index, index) -> index
          %30 = "emitc.subscript"(%0, %23) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
          %31 = "emitc.apply"(%30) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
          %32 = "emitc.call_opaque"(%31, %17, %16) <{callee = "__riscv_vlse8_v_i8m1"}> : (!emitc.ptr<i8>, index, index) -> !emitc.opaque<"vint8m1_t">
          %33 = "emitc.subscript"(%0, %27) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
          %34 = "emitc.apply"(%33) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
          %35 = "emitc.call_opaque"(%34, %17, %16) <{callee = "__riscv_vlse8_v_i8m1"}> : (!emitc.ptr<i8>, index, index) -> !emitc.opaque<"vint8m1_t">
          %36 = "emitc.subscript"(%0, %28) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
          %38 = "emitc.call_opaque"(%37, %17, %16) <{callee = "__riscv_vlse8_v_i8m1"}> : (!emitc.ptr<i8>, index, index) -> !emitc.opaque<"vint8m1_t">
          %39 = "emitc.subscript"(%0, %29) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
          %40 = "emitc.apply"(%39) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
          %41 = "emitc.call_opaque"(%40, %17, %16) <{callee = "__riscv_vlse8_v_i8m1"}> : (!emitc.ptr<i8>, index, index) -> !emitc.opaque<"vint8m1_t">
          %42 = "emitc.constant"() {value = 31 : index} : () -> index
          %43 = "emitc.add"(%2, %42) : (index, index) -> index
          %44 = "emitc.div"(%43, %9) : (index, index) -> index
          %45 = "emitc.mul"(%44, %9) : (index, index) -> index
          %46 = "emitc.div"(%10, %8) : (index, index) -> index
          %47 = "emitc.mul"(%45, %8) : (index, index) -> index
          %48 = "emitc.mul"(%46, %47) : (index, index) -> index
          %49 = "emitc.mul"(%13, %8) : (index, index) -> index
          %50 = "emitc.mul"(%20, %8) : (index, index) -> index
          %51 = "emitc.add"(%48, %49) : (index, index) -> index
          %52 = "emitc.add"(%51, %50) : (index, index) -> index
          %53 = "emitc.add"(%52, %24) : (index, index) -> index
          %54 = "emitc.add"(%52, %25) : (index, index) -> index
          %55 = "emitc.add"(%52, %26) : (index, index) -> index
          %56 = "emitc.subscript"(%1, %52) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
          %57 = "emitc.apply"(%56) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
          "emitc.call_opaque"(%57, %7, %32, %16) <{callee = "__riscv_vsse8_v_i8m1"}> : (!emitc.ptr<i8>, index, !emitc.opaque<"vint8m1_t">, index) -> ()
          %58 = "emitc.subscript"(%1, %53) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
          %59 = "emitc.apply"(%58) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
          "emitc.call_opaque"(%59, %7, %35, %16) <{callee = "__riscv_vsse8_v_i8m1"}> : (!emitc.ptr<i8>, index, !emitc.opaque<"vint8m1_t">, index) -> ()
          %60 = "emitc.subscript"(%1, %54) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
          %61 = "emitc.apply"(%60) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
          "emitc.call_opaque"(%61, %7, %38, %16) <{callee = "__riscv_vsse8_v_i8m1"}> : (!emitc.ptr<i8>, index, !emitc.opaque<"vint8m1_t">, index) -> ()
          %62 = "emitc.subscript"(%1, %55) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
          %63 = "emitc.apply"(%62) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
          "emitc.call_opaque"(%63, %7, %41, %16) <{callee = "__riscv_vsse8_v_i8m1"}> : (!emitc.ptr<i8>, index, !emitc.opaque<"vint8m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<i8>, !emitc.ptr<i8>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_DOT_INT8_LANE_Pack_i8_RR8", function_type = (!emitc.ptr<i8>, !emitc.ptr<i8>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<i8>, %1 : !emitc.ptr<i8>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<i8>, !emitc.ptr<i8>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_DOT_INT8_LANE_Pack_i8_RR8", function_type = (!emitc.ptr<i8>, !emitc.ptr<i8>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<i8>, %1 : !emitc.ptr<i8>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 8 : index} : () -> index
    %8 = "emitc.constant"() {value = 32 : index} : () -> index
    %9 = "emitc.constant"() {value = 1 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%10 : index):
      %11 = "emitc.sub"(%2, %10) : (index, index) -> index
      %12 = "emitc.call_opaque"(%11, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %8) ({
      ^bb2(%13 : index):
        %14 = "emitc.sub"(%3, %13) : (index, index) -> index
        %15 = "emitc.call_opaque"(%14, %8) <{callee = "std::min"}> : (index, index) -> index
        %16 = "emitc.call_opaque"(%15) <{callee = "__riscv_vsetvl_e8m1"}> : (index) -> index
        %17 = "emitc.mul"(%4, %9) : (index, index) -> index
        "emitc.for"(%5, %12, %6) ({
        ^bb3(%18 : index):
          %19 = "emitc.add"(%10, %18) : (index, index) -> index
          %20 = "emitc.mul"(%13, %4) : (index, index) -> index
          %21 = "emitc.add"(%20, %19) : (index, index) -> index
          %22 = "emitc.subscript"(%0, %21) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
          %23 = "emitc.apply"(%22) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
          %24 = "emitc.call_opaque"(%23, %17, %16) <{callee = "__riscv_vlse8_v_i8m1"}> : (!emitc.ptr<i8>, index, index) -> !emitc.opaque<"vint8m1_t">
          %25 = "emitc.constant"() {value = 31 : index} : () -> index
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
          %36 = "emitc.subscript"(%1, %35) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
          %37 = "emitc.apply"(%36) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
          "emitc.call_opaque"(%37, %24, %16) <{callee = "__riscv_vse8_v_i8m1"}> : (!emitc.ptr<i8>, !emitc.opaque<"vint8m1_t">, index) -> ()
          "emitc.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<i8>, !emitc.ptr<i8>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_DOT_INT8_LANE_Pack_i8_RR8", function_type = (!emitc.ptr<i8>, !emitc.ptr<i8>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<i8>, %1 : !emitc.ptr<i8>, %2 : index, %3 : index, %4 : index):
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<i8>, !emitc.ptr<i8>, index, index, index) -> index} : () -> ()
}) : () -> ()
