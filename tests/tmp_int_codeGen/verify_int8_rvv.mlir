"builtin.module"() ({
  "func.func"() <{sym_name = "uk_riscv1_gemm_Ccol_32x6_vl256_m4_int8_int32", function_type = (i32, i32, index, !emitc.ptr<i8>, !emitc.ptr<i8>, !emitc.ptr<i32>, i32, index) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : index, %3 : !emitc.ptr<i8>, %4 : !emitc.ptr<i8>, %5 : !emitc.ptr<i32>, %6 : i32, %7 : index):
    %8 = "emitc.constant"() {value = 32 : index} : () -> index
    %9 = "emitc.constant"() {value = 0 : i32} : () -> i32
    %10 = "emitc.constant"() {value = 0 : index} : () -> index
    %11 = "emitc.constant"() {value = 1 : index} : () -> index
    %12 = "emitc.constant"() {value = 32 : index} : () -> index
    %13 = "emitc.constant"() {value = 6 : index} : () -> index
    %14 = "emitc.constant"() {value = 0 : i8} : () -> i8
    %15 = "emitc.call_opaque"(%9, %8) <{callee = "__riscv_vmv_v_x_i32m4"}> : (i32, index) -> !emitc.opaque<"vint32m4_t">
    %16 = "emitc.call_opaque"(%9, %8) <{callee = "__riscv_vmv_v_x_i32m4"}> : (i32, index) -> !emitc.opaque<"vint32m4_t">
    %17 = "emitc.call_opaque"(%9, %8) <{callee = "__riscv_vmv_v_x_i32m4"}> : (i32, index) -> !emitc.opaque<"vint32m4_t">
    %18 = "emitc.call_opaque"(%9, %8) <{callee = "__riscv_vmv_v_x_i32m4"}> : (i32, index) -> !emitc.opaque<"vint32m4_t">
    %19 = "emitc.call_opaque"(%9, %8) <{callee = "__riscv_vmv_v_x_i32m4"}> : (i32, index) -> !emitc.opaque<"vint32m4_t">
    %20 = "emitc.call_opaque"(%9, %8) <{callee = "__riscv_vmv_v_x_i32m4"}> : (i32, index) -> !emitc.opaque<"vint32m4_t">
    %21 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m4_t">>
    "emitc.assign"(%21, %15) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
    %22 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m4_t">>
    "emitc.assign"(%22, %16) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
    %23 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m4_t">>
    "emitc.assign"(%23, %17) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
    %24 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m4_t">>
    "emitc.assign"(%24, %18) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
    %25 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m4_t">>
    "emitc.assign"(%25, %19) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
    %26 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m4_t">>
    "emitc.assign"(%26, %20) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
    %27 = "emitc.variable"() {value = 0 : index} : () -> !emitc.lvalue<index>
    "emitc.assign"(%27, %10) : (!emitc.lvalue<index>, index) -> ()
    %28 = "emitc.variable"() {value = 0 : index} : () -> !emitc.lvalue<index>
    "emitc.assign"(%28, %10) : (!emitc.lvalue<index>, index) -> ()
    "emitc.for"(%10, %2, %11) ({
    ^bb1(%29 : index):
      %30 = "emitc.load"(%27) : (!emitc.lvalue<index>) -> index
      %31 = "emitc.subscript"(%3, %30) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %32 = "emitc.apply"(%31) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
      %33 = "emitc.call_opaque"(%32, %8) <{callee = "__riscv_vle8_v_i8m1"}> : (!emitc.ptr<i8>, index) -> !emitc.opaque<"vint8m1_t">
      %34 = "emitc.call_opaque"(%33, %14, %8) <{callee = "__riscv_vwadd_vx_i16m2"}> : (!emitc.opaque<"vint8m1_t">, i8, index) -> !emitc.opaque<"vint16m2_t">
      %35 = "emitc.constant"() {value = 0 : index} : () -> index
      %36 = "emitc.load"(%28) : (!emitc.lvalue<index>) -> index
      %37 = "emitc.add"(%36, %35) : (index, index) -> index
      %38 = "emitc.subscript"(%4, %37) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %39 = "emitc.load"(%38) : (!emitc.lvalue<i8>) -> i8
      %40 = "emitc.load"(%21) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
      %41 = "emitc.call_opaque"(%40, %39, %34, %8) <{callee = "__riscv_vwmacc_vx_i32m4"}> : (!emitc.opaque<"vint32m4_t">, i8, !emitc.opaque<"vint16m2_t">, index) -> !emitc.opaque<"vint32m4_t">
      %42 = "emitc.constant"() {value = 1 : index} : () -> index
      %43 = "emitc.add"(%36, %42) : (index, index) -> index
      %44 = "emitc.subscript"(%4, %43) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %45 = "emitc.load"(%44) : (!emitc.lvalue<i8>) -> i8
      %46 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
      %47 = "emitc.call_opaque"(%46, %45, %34, %8) <{callee = "__riscv_vwmacc_vx_i32m4"}> : (!emitc.opaque<"vint32m4_t">, i8, !emitc.opaque<"vint16m2_t">, index) -> !emitc.opaque<"vint32m4_t">
      %48 = "emitc.constant"() {value = 2 : index} : () -> index
      %49 = "emitc.add"(%36, %48) : (index, index) -> index
      %50 = "emitc.subscript"(%4, %49) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %51 = "emitc.load"(%50) : (!emitc.lvalue<i8>) -> i8
      %52 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
      %53 = "emitc.call_opaque"(%52, %51, %34, %8) <{callee = "__riscv_vwmacc_vx_i32m4"}> : (!emitc.opaque<"vint32m4_t">, i8, !emitc.opaque<"vint16m2_t">, index) -> !emitc.opaque<"vint32m4_t">
      %54 = "emitc.constant"() {value = 3 : index} : () -> index
      %55 = "emitc.add"(%36, %54) : (index, index) -> index
      %56 = "emitc.subscript"(%4, %55) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %57 = "emitc.load"(%56) : (!emitc.lvalue<i8>) -> i8
      %58 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
      %59 = "emitc.call_opaque"(%58, %57, %34, %8) <{callee = "__riscv_vwmacc_vx_i32m4"}> : (!emitc.opaque<"vint32m4_t">, i8, !emitc.opaque<"vint16m2_t">, index) -> !emitc.opaque<"vint32m4_t">
      %60 = "emitc.constant"() {value = 4 : index} : () -> index
      %61 = "emitc.add"(%36, %60) : (index, index) -> index
      %62 = "emitc.subscript"(%4, %61) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %63 = "emitc.load"(%62) : (!emitc.lvalue<i8>) -> i8
      %64 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
      %65 = "emitc.call_opaque"(%64, %63, %34, %8) <{callee = "__riscv_vwmacc_vx_i32m4"}> : (!emitc.opaque<"vint32m4_t">, i8, !emitc.opaque<"vint16m2_t">, index) -> !emitc.opaque<"vint32m4_t">
      %66 = "emitc.constant"() {value = 5 : index} : () -> index
      %67 = "emitc.add"(%36, %66) : (index, index) -> index
      %68 = "emitc.subscript"(%4, %67) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %69 = "emitc.load"(%68) : (!emitc.lvalue<i8>) -> i8
      %70 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
      %71 = "emitc.call_opaque"(%70, %69, %34, %8) <{callee = "__riscv_vwmacc_vx_i32m4"}> : (!emitc.opaque<"vint32m4_t">, i8, !emitc.opaque<"vint16m2_t">, index) -> !emitc.opaque<"vint32m4_t">
      %72 = "emitc.add"(%30, %12) : (index, index) -> index
      %73 = "emitc.add"(%36, %13) : (index, index) -> index
      "emitc.assign"(%21, %41) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
      "emitc.assign"(%22, %47) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
      "emitc.assign"(%23, %53) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
      "emitc.assign"(%24, %59) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
      "emitc.assign"(%25, %65) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
      "emitc.assign"(%26, %71) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>, !emitc.opaque<"vint32m4_t">) -> ()
      "emitc.assign"(%27, %72) : (!emitc.lvalue<index>, index) -> ()
      "emitc.assign"(%28, %73) : (!emitc.lvalue<index>, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %74 = "emitc.load"(%21) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
    %75 = "emitc.load"(%22) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
    %76 = "emitc.load"(%23) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
    %77 = "emitc.load"(%24) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
    %78 = "emitc.load"(%25) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
    %79 = "emitc.load"(%26) : (!emitc.lvalue<!emitc.opaque<"vint32m4_t">>) -> !emitc.opaque<"vint32m4_t">
    %80 = "emitc.load"(%27) : (!emitc.lvalue<index>) -> index
    %81 = "emitc.load"(%28) : (!emitc.lvalue<index>) -> index
    %82 = "emitc.constant"() {value = 0 : index} : () -> index
    %83 = "emitc.mul"(%82, %7) : (index, index) -> index
    %84 = "emitc.subscript"(%5, %83) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %85 = "emitc.apply"(%84) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%85, %74, %8) <{callee = "__riscv_vse32_v_i32m4"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m4_t">, index) -> ()
    %86 = "emitc.constant"() {value = 1 : index} : () -> index
    %87 = "emitc.mul"(%86, %7) : (index, index) -> index
    %88 = "emitc.subscript"(%5, %87) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %89 = "emitc.apply"(%88) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%89, %75, %8) <{callee = "__riscv_vse32_v_i32m4"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m4_t">, index) -> ()
    %90 = "emitc.constant"() {value = 2 : index} : () -> index
    %91 = "emitc.mul"(%90, %7) : (index, index) -> index
    %92 = "emitc.subscript"(%5, %91) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %93 = "emitc.apply"(%92) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%93, %76, %8) <{callee = "__riscv_vse32_v_i32m4"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m4_t">, index) -> ()
    %94 = "emitc.constant"() {value = 3 : index} : () -> index
    %95 = "emitc.mul"(%94, %7) : (index, index) -> index
    %96 = "emitc.subscript"(%5, %95) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %97 = "emitc.apply"(%96) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%97, %77, %8) <{callee = "__riscv_vse32_v_i32m4"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m4_t">, index) -> ()
    %98 = "emitc.constant"() {value = 4 : index} : () -> index
    %99 = "emitc.mul"(%98, %7) : (index, index) -> index
    %100 = "emitc.subscript"(%5, %99) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %101 = "emitc.apply"(%100) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%101, %78, %8) <{callee = "__riscv_vse32_v_i32m4"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m4_t">, index) -> ()
    %102 = "emitc.constant"() {value = 5 : index} : () -> index
    %103 = "emitc.mul"(%102, %7) : (index, index) -> index
    %104 = "emitc.subscript"(%5, %103) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %105 = "emitc.apply"(%104) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%105, %79, %8) <{callee = "__riscv_vse32_v_i32m4"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m4_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32, i32, index, !emitc.ptr<i8>, !emitc.ptr<i8>, !emitc.ptr<i32>, i32, index) -> index} : () -> ()
}) : () -> ()
