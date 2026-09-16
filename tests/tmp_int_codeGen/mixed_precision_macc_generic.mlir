"builtin.module"() ({
  "func.func"() <{sym_name = "xdsl_api_gemm_RVV_8x8_i8i8i32_b0", function_type = (index, !emitc.ptr<i8>, index, !emitc.ptr<i8>, index, !emitc.ptr<i32>, index) -> ()}> ({
  ^bb0(%0 : index, %1 : !emitc.ptr<i8>, %2 : index, %3 : !emitc.ptr<i8>, %4 : index, %5 : !emitc.ptr<i32>, %6 : index):
    %7 = "emitc.constant"() {value = 0 : index} : () -> index
    %8 = "emitc.constant"() {value = 1 : index} : () -> index
    %9 = "emitc.constant"() {value = 8 : index} : () -> index
    %10 = "emitc.constant"() {value = 0 : i8} : () -> i8
    %11 = "emitc.constant"() {value = 0 : i16} : () -> i16
    %12 = "emitc.constant"() {value = 0 : i32} : () -> i32
    %13 = "emitc.call_opaque"(%9) <{callee = "__riscv_vsetvl_e8mf4"}> : (index) -> index
    %14 = "emitc.constant"() {value = 0 : index} : () -> index
    %15 = "emitc.constant"() {value = 1 : index} : () -> index
    %16 = "emitc.constant"() {value = 2 : index} : () -> index
    %17 = "emitc.constant"() {value = 3 : index} : () -> index
    %18 = "emitc.constant"() {value = 4 : index} : () -> index
    %19 = "emitc.constant"() {value = 5 : index} : () -> index
    %20 = "emitc.constant"() {value = 6 : index} : () -> index
    %21 = "emitc.constant"() {value = 7 : index} : () -> index
    %22 = "emitc.call_opaque"(%12, %13) <{callee = "__riscv_vmv_v_x_i32m1"}> : (i32, index) -> !emitc.opaque<"vint32m1_t">
    %23 = "emitc.call_opaque"(%12, %13) <{callee = "__riscv_vmv_v_x_i32m1"}> : (i32, index) -> !emitc.opaque<"vint32m1_t">
    %24 = "emitc.call_opaque"(%12, %13) <{callee = "__riscv_vmv_v_x_i32m1"}> : (i32, index) -> !emitc.opaque<"vint32m1_t">
    %25 = "emitc.call_opaque"(%12, %13) <{callee = "__riscv_vmv_v_x_i32m1"}> : (i32, index) -> !emitc.opaque<"vint32m1_t">
    %26 = "emitc.call_opaque"(%12, %13) <{callee = "__riscv_vmv_v_x_i32m1"}> : (i32, index) -> !emitc.opaque<"vint32m1_t">
    %27 = "emitc.call_opaque"(%12, %13) <{callee = "__riscv_vmv_v_x_i32m1"}> : (i32, index) -> !emitc.opaque<"vint32m1_t">
    %28 = "emitc.call_opaque"(%12, %13) <{callee = "__riscv_vmv_v_x_i32m1"}> : (i32, index) -> !emitc.opaque<"vint32m1_t">
    %29 = "emitc.call_opaque"(%12, %13) <{callee = "__riscv_vmv_v_x_i32m1"}> : (i32, index) -> !emitc.opaque<"vint32m1_t">
    %30 = "emitc.call_opaque"(%11, %13) <{callee = "__riscv_vmv_v_x_i16mf2"}> : (i16, index) -> !emitc.opaque<"vint16mf2_t">
    %31 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
    "emitc.assign"(%31, %22) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
    %32 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
    "emitc.assign"(%32, %23) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
    %33 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
    "emitc.assign"(%33, %24) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
    %34 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
    "emitc.assign"(%34, %25) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
    %35 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
    "emitc.assign"(%35, %26) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
    %36 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
    "emitc.assign"(%36, %27) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
    %37 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
    "emitc.assign"(%37, %28) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
    %38 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
    "emitc.assign"(%38, %29) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
    %39 = "emitc.variable"() {value = 0 : index} : () -> !emitc.lvalue<index>
    "emitc.assign"(%39, %7) : (!emitc.lvalue<index>, index) -> ()
    %40 = "emitc.variable"() {value = 0 : index} : () -> !emitc.lvalue<index>
    "emitc.assign"(%40, %7) : (!emitc.lvalue<index>, index) -> ()
    "emitc.for"(%7, %0, %8) ({
    ^bb1(%41 : index):
      %42 = "emitc.load"(%39) : (!emitc.lvalue<index>) -> index
      %43 = "emitc.subscript"(%1, %42) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %44 = "emitc.apply"(%43) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
      %45 = "emitc.call_opaque"(%44, %13) <{callee = "__riscv_vle8_v_i8mf4"}> : (!emitc.ptr<i8>, index) -> !emitc.opaque<"vint8mf4_t">
      %46 = "emitc.load"(%40) : (!emitc.lvalue<index>) -> index
      %47 = "emitc.add"(%46, %14) : (index, index) -> index
      %48 = "emitc.subscript"(%3, %47) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %49 = "emitc.load"(%48) : (!emitc.lvalue<i8>) -> i8
      %50 = "emitc.call_opaque"(%30, %49, %45, %13) <{callee = "__riscv_vwmacc_vx_i16mf2"}> : (!emitc.opaque<"vint16mf2_t">, i8, !emitc.opaque<"vint8mf4_t">, index) -> !emitc.opaque<"vint16mf2_t">
      %51 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
      %52 = "emitc.call_opaque"(%51, %50, %13) <{callee = "__riscv_vwadd_wv_i32m1"}> : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint16mf2_t">, index) -> !emitc.opaque<"vint32m1_t">
      %53 = "emitc.add"(%46, %15) : (index, index) -> index
      %54 = "emitc.subscript"(%3, %53) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %55 = "emitc.load"(%54) : (!emitc.lvalue<i8>) -> i8
      %56 = "emitc.call_opaque"(%30, %55, %45, %13) <{callee = "__riscv_vwmacc_vx_i16mf2"}> : (!emitc.opaque<"vint16mf2_t">, i8, !emitc.opaque<"vint8mf4_t">, index) -> !emitc.opaque<"vint16mf2_t">
      %57 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
      %58 = "emitc.call_opaque"(%57, %56, %13) <{callee = "__riscv_vwadd_wv_i32m1"}> : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint16mf2_t">, index) -> !emitc.opaque<"vint32m1_t">
      %59 = "emitc.add"(%46, %16) : (index, index) -> index
      %60 = "emitc.subscript"(%3, %59) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %61 = "emitc.load"(%60) : (!emitc.lvalue<i8>) -> i8
      %62 = "emitc.call_opaque"(%30, %61, %45, %13) <{callee = "__riscv_vwmacc_vx_i16mf2"}> : (!emitc.opaque<"vint16mf2_t">, i8, !emitc.opaque<"vint8mf4_t">, index) -> !emitc.opaque<"vint16mf2_t">
      %63 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
      %64 = "emitc.call_opaque"(%63, %62, %13) <{callee = "__riscv_vwadd_wv_i32m1"}> : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint16mf2_t">, index) -> !emitc.opaque<"vint32m1_t">
      %65 = "emitc.add"(%46, %17) : (index, index) -> index
      %66 = "emitc.subscript"(%3, %65) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %67 = "emitc.load"(%66) : (!emitc.lvalue<i8>) -> i8
      %68 = "emitc.call_opaque"(%30, %67, %45, %13) <{callee = "__riscv_vwmacc_vx_i16mf2"}> : (!emitc.opaque<"vint16mf2_t">, i8, !emitc.opaque<"vint8mf4_t">, index) -> !emitc.opaque<"vint16mf2_t">
      %69 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
      %70 = "emitc.call_opaque"(%69, %68, %13) <{callee = "__riscv_vwadd_wv_i32m1"}> : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint16mf2_t">, index) -> !emitc.opaque<"vint32m1_t">
      %71 = "emitc.add"(%46, %18) : (index, index) -> index
      %72 = "emitc.subscript"(%3, %71) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %73 = "emitc.load"(%72) : (!emitc.lvalue<i8>) -> i8
      %74 = "emitc.call_opaque"(%30, %73, %45, %13) <{callee = "__riscv_vwmacc_vx_i16mf2"}> : (!emitc.opaque<"vint16mf2_t">, i8, !emitc.opaque<"vint8mf4_t">, index) -> !emitc.opaque<"vint16mf2_t">
      %75 = "emitc.load"(%35) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
      %76 = "emitc.call_opaque"(%75, %74, %13) <{callee = "__riscv_vwadd_wv_i32m1"}> : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint16mf2_t">, index) -> !emitc.opaque<"vint32m1_t">
      %77 = "emitc.add"(%46, %19) : (index, index) -> index
      %78 = "emitc.subscript"(%3, %77) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %79 = "emitc.load"(%78) : (!emitc.lvalue<i8>) -> i8
      %80 = "emitc.call_opaque"(%30, %79, %45, %13) <{callee = "__riscv_vwmacc_vx_i16mf2"}> : (!emitc.opaque<"vint16mf2_t">, i8, !emitc.opaque<"vint8mf4_t">, index) -> !emitc.opaque<"vint16mf2_t">
      %81 = "emitc.load"(%36) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
      %82 = "emitc.call_opaque"(%81, %80, %13) <{callee = "__riscv_vwadd_wv_i32m1"}> : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint16mf2_t">, index) -> !emitc.opaque<"vint32m1_t">
      %83 = "emitc.add"(%46, %20) : (index, index) -> index
      %84 = "emitc.subscript"(%3, %83) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %85 = "emitc.load"(%84) : (!emitc.lvalue<i8>) -> i8
      %86 = "emitc.call_opaque"(%30, %85, %45, %13) <{callee = "__riscv_vwmacc_vx_i16mf2"}> : (!emitc.opaque<"vint16mf2_t">, i8, !emitc.opaque<"vint8mf4_t">, index) -> !emitc.opaque<"vint16mf2_t">
      %87 = "emitc.load"(%37) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
      %88 = "emitc.call_opaque"(%87, %86, %13) <{callee = "__riscv_vwadd_wv_i32m1"}> : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint16mf2_t">, index) -> !emitc.opaque<"vint32m1_t">
      %89 = "emitc.add"(%46, %21) : (index, index) -> index
      %90 = "emitc.subscript"(%3, %89) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
      %91 = "emitc.load"(%90) : (!emitc.lvalue<i8>) -> i8
      %92 = "emitc.call_opaque"(%30, %91, %45, %13) <{callee = "__riscv_vwmacc_vx_i16mf2"}> : (!emitc.opaque<"vint16mf2_t">, i8, !emitc.opaque<"vint8mf4_t">, index) -> !emitc.opaque<"vint16mf2_t">
      %93 = "emitc.load"(%38) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
      %94 = "emitc.call_opaque"(%93, %92, %13) <{callee = "__riscv_vwadd_wv_i32m1"}> : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint16mf2_t">, index) -> !emitc.opaque<"vint32m1_t">
      %95 = "emitc.add"(%42, %2) : (index, index) -> index
      %96 = "emitc.add"(%46, %4) : (index, index) -> index
      "emitc.assign"(%31, %52) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
      "emitc.assign"(%32, %58) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
      "emitc.assign"(%33, %64) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
      "emitc.assign"(%34, %70) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
      "emitc.assign"(%35, %76) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
      "emitc.assign"(%36, %82) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
      "emitc.assign"(%37, %88) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
      "emitc.assign"(%38, %94) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>, !emitc.opaque<"vint32m1_t">) -> ()
      "emitc.assign"(%39, %95) : (!emitc.lvalue<index>, index) -> ()
      "emitc.assign"(%40, %96) : (!emitc.lvalue<index>, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %97 = "emitc.load"(%31) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
    %98 = "emitc.load"(%32) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
    %99 = "emitc.load"(%33) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
    %100 = "emitc.load"(%34) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
    %101 = "emitc.load"(%35) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
    %102 = "emitc.load"(%36) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
    %103 = "emitc.load"(%37) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
    %104 = "emitc.load"(%38) : (!emitc.lvalue<!emitc.opaque<"vint32m1_t">>) -> !emitc.opaque<"vint32m1_t">
    %105 = "emitc.load"(%39) : (!emitc.lvalue<index>) -> index
    %106 = "emitc.load"(%40) : (!emitc.lvalue<index>) -> index
    %107 = "emitc.mul"(%14, %6) : (index, index) -> index
    %108 = "emitc.subscript"(%5, %107) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %109 = "emitc.apply"(%108) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%109, %97, %13) <{callee = "__riscv_vse32_v_i32m1"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m1_t">, index) -> ()
    %110 = "emitc.mul"(%15, %6) : (index, index) -> index
    %111 = "emitc.subscript"(%5, %110) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %112 = "emitc.apply"(%111) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%112, %98, %13) <{callee = "__riscv_vse32_v_i32m1"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m1_t">, index) -> ()
    %113 = "emitc.mul"(%16, %6) : (index, index) -> index
    %114 = "emitc.subscript"(%5, %113) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %115 = "emitc.apply"(%114) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%115, %99, %13) <{callee = "__riscv_vse32_v_i32m1"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m1_t">, index) -> ()
    %116 = "emitc.mul"(%17, %6) : (index, index) -> index
    %117 = "emitc.subscript"(%5, %116) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %118 = "emitc.apply"(%117) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%118, %100, %13) <{callee = "__riscv_vse32_v_i32m1"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m1_t">, index) -> ()
    %119 = "emitc.mul"(%18, %6) : (index, index) -> index
    %120 = "emitc.subscript"(%5, %119) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %121 = "emitc.apply"(%120) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%121, %101, %13) <{callee = "__riscv_vse32_v_i32m1"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m1_t">, index) -> ()
    %122 = "emitc.mul"(%19, %6) : (index, index) -> index
    %123 = "emitc.subscript"(%5, %122) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %124 = "emitc.apply"(%123) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%124, %102, %13) <{callee = "__riscv_vse32_v_i32m1"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m1_t">, index) -> ()
    %125 = "emitc.mul"(%20, %6) : (index, index) -> index
    %126 = "emitc.subscript"(%5, %125) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %127 = "emitc.apply"(%126) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%127, %103, %13) <{callee = "__riscv_vse32_v_i32m1"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m1_t">, index) -> ()
    %128 = "emitc.mul"(%21, %6) : (index, index) -> index
    %129 = "emitc.subscript"(%5, %128) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %130 = "emitc.apply"(%129) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%130, %104, %13) <{callee = "__riscv_vse32_v_i32m1"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<i8>, index, !emitc.ptr<i8>, index, !emitc.ptr<i32>, index) -> index} : () -> ()
}) : () -> ()