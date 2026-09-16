"builtin.module"() ({
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_32x6_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
    %19 = "emitc.constant"() {value = 9 : index} : () -> index
    %20 = "emitc.constant"() {value = 10 : index} : () -> index
    %21 = "emitc.constant"() {value = 11 : index} : () -> index
    %22 = "emitc.constant"() {value = 12 : index} : () -> index
    %23 = "emitc.constant"() {value = 13 : index} : () -> index
    %24 = "emitc.constant"() {value = 14 : index} : () -> index
    %25 = "emitc.constant"() {value = 15 : index} : () -> index
    %26 = "emitc.constant"() {value = 16 : index} : () -> index
    %27 = "emitc.constant"() {value = 17 : index} : () -> index
    %28 = "emitc.constant"() {value = 18 : index} : () -> index
    %29 = "emitc.constant"() {value = 19 : index} : () -> index
    %30 = "emitc.constant"() {value = 20 : index} : () -> index
    %31 = "emitc.constant"() {value = 21 : index} : () -> index
    %32 = "emitc.constant"() {value = 22 : index} : () -> index
    %33 = "emitc.constant"() {value = 23 : index} : () -> index
    %34 = "emitc.constant"() {value = 24 : index} : () -> index
    %35 = "emitc.constant"() {value = 25 : index} : () -> index
    %36 = "emitc.constant"() {value = 26 : index} : () -> index
    %37 = "emitc.constant"() {value = 27 : index} : () -> index
    %38 = "emitc.constant"() {value = 28 : index} : () -> index
    %39 = "emitc.constant"() {value = 29 : index} : () -> index
    %40 = "emitc.constant"() {value = 30 : index} : () -> index
    %41 = "emitc.constant"() {value = 31 : index} : () -> index
    %42 = "emitc.constant"() {value = 32 : index} : () -> index
    %43 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %44 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %46 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %47 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %48 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %49 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %50 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %51 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %52 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %53 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %54 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %55 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %56 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %57 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %58 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %59 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %60 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %61 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %62 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %63 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %64 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %66 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %67 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%67, %43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %68 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%68, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %69 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%69, %45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %70 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%70, %46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %71 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%71, %47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %72 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%72, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %73 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%73, %49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %74 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%74, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %75 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%75, %51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %76 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%76, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %77 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%77, %53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %78 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%78, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %79 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%79, %55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %80 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%80, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %81 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%81, %57) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %82 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%82, %58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %83 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%83, %59) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %84 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%84, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %85 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%85, %61) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %86 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%86, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %87 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%87, %63) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %88 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%88, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %89 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%89, %65) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %90 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%90, %66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%91 : index):
      %92 = "emitc.mul"(%91, %3) : (index, index) -> index
      %93 = "emitc.constant"() {value = 8 : index} : () -> index
      %94 = "emitc.subscript"(%2, %92) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %96 = "emitc.call_opaque"(%95, %93) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %97 = "emitc.add"(%92, %18) : (index, index) -> index
      %98 = "emitc.subscript"(%2, %97) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %99 = "emitc.apply"(%98) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %100 = "emitc.call_opaque"(%99, %93) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %101 = "emitc.add"(%92, %26) : (index, index) -> index
      %102 = "emitc.subscript"(%2, %101) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %103 = "emitc.apply"(%102) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %104 = "emitc.call_opaque"(%103, %93) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %105 = "emitc.add"(%92, %34) : (index, index) -> index
      %106 = "emitc.subscript"(%2, %105) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %107 = "emitc.apply"(%106) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %108 = "emitc.call_opaque"(%107, %93) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %109 = "emitc.mul"(%91, %5) : (index, index) -> index
      %110 = "emitc.add"(%109, %11) : (index, index) -> index
      %111 = "emitc.add"(%109, %12) : (index, index) -> index
      %112 = "emitc.add"(%109, %13) : (index, index) -> index
      %113 = "emitc.add"(%109, %14) : (index, index) -> index
      %114 = "emitc.add"(%109, %15) : (index, index) -> index
      %115 = "emitc.load"(%67) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %116 = "emitc.subscript"(%4, %109) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %117 = "emitc.load"(%116) : (!emitc.lvalue<f32>) -> f32
      %118 = "emitc.call_opaque"(%115, %117, %96, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %119 = "emitc.load"(%68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %120 = "emitc.subscript"(%4, %109) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %121 = "emitc.load"(%120) : (!emitc.lvalue<f32>) -> f32
      %122 = "emitc.call_opaque"(%119, %121, %100, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %123 = "emitc.load"(%69) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %124 = "emitc.subscript"(%4, %109) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %125 = "emitc.load"(%124) : (!emitc.lvalue<f32>) -> f32
      %126 = "emitc.call_opaque"(%123, %125, %104, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %127 = "emitc.load"(%70) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %128 = "emitc.subscript"(%4, %109) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %129 = "emitc.load"(%128) : (!emitc.lvalue<f32>) -> f32
      %130 = "emitc.call_opaque"(%127, %129, %108, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %131 = "emitc.load"(%71) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %132 = "emitc.subscript"(%4, %110) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %133 = "emitc.load"(%132) : (!emitc.lvalue<f32>) -> f32
      %134 = "emitc.call_opaque"(%131, %133, %96, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %135 = "emitc.load"(%72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %136 = "emitc.subscript"(%4, %110) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %137 = "emitc.load"(%136) : (!emitc.lvalue<f32>) -> f32
      %138 = "emitc.call_opaque"(%135, %137, %100, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %139 = "emitc.load"(%73) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %140 = "emitc.subscript"(%4, %110) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %141 = "emitc.load"(%140) : (!emitc.lvalue<f32>) -> f32
      %142 = "emitc.call_opaque"(%139, %141, %104, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %143 = "emitc.load"(%74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %144 = "emitc.subscript"(%4, %110) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %145 = "emitc.load"(%144) : (!emitc.lvalue<f32>) -> f32
      %146 = "emitc.call_opaque"(%143, %145, %108, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %147 = "emitc.load"(%75) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %148 = "emitc.subscript"(%4, %111) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %149 = "emitc.load"(%148) : (!emitc.lvalue<f32>) -> f32
      %150 = "emitc.call_opaque"(%147, %149, %96, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %151 = "emitc.load"(%76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %152 = "emitc.subscript"(%4, %111) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %153 = "emitc.load"(%152) : (!emitc.lvalue<f32>) -> f32
      %154 = "emitc.call_opaque"(%151, %153, %100, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %155 = "emitc.load"(%77) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %156 = "emitc.subscript"(%4, %111) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %157 = "emitc.load"(%156) : (!emitc.lvalue<f32>) -> f32
      %158 = "emitc.call_opaque"(%155, %157, %104, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %159 = "emitc.load"(%78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %160 = "emitc.subscript"(%4, %111) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %161 = "emitc.load"(%160) : (!emitc.lvalue<f32>) -> f32
      %162 = "emitc.call_opaque"(%159, %161, %108, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %163 = "emitc.load"(%79) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %164 = "emitc.subscript"(%4, %112) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %165 = "emitc.load"(%164) : (!emitc.lvalue<f32>) -> f32
      %166 = "emitc.call_opaque"(%163, %165, %96, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %167 = "emitc.load"(%80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %168 = "emitc.subscript"(%4, %112) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %169 = "emitc.load"(%168) : (!emitc.lvalue<f32>) -> f32
      %170 = "emitc.call_opaque"(%167, %169, %100, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %171 = "emitc.load"(%81) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %172 = "emitc.subscript"(%4, %112) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %173 = "emitc.load"(%172) : (!emitc.lvalue<f32>) -> f32
      %174 = "emitc.call_opaque"(%171, %173, %104, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %175 = "emitc.load"(%82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %176 = "emitc.subscript"(%4, %112) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %177 = "emitc.load"(%176) : (!emitc.lvalue<f32>) -> f32
      %178 = "emitc.call_opaque"(%175, %177, %108, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %179 = "emitc.load"(%83) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %180 = "emitc.subscript"(%4, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %181 = "emitc.load"(%180) : (!emitc.lvalue<f32>) -> f32
      %182 = "emitc.call_opaque"(%179, %181, %96, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %183 = "emitc.load"(%84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %184 = "emitc.subscript"(%4, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %185 = "emitc.load"(%184) : (!emitc.lvalue<f32>) -> f32
      %186 = "emitc.call_opaque"(%183, %185, %100, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %187 = "emitc.load"(%85) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %188 = "emitc.subscript"(%4, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %189 = "emitc.load"(%188) : (!emitc.lvalue<f32>) -> f32
      %190 = "emitc.call_opaque"(%187, %189, %104, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %191 = "emitc.load"(%86) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %192 = "emitc.subscript"(%4, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %193 = "emitc.load"(%192) : (!emitc.lvalue<f32>) -> f32
      %194 = "emitc.call_opaque"(%191, %193, %108, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %195 = "emitc.load"(%87) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %196 = "emitc.subscript"(%4, %114) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %197 = "emitc.load"(%196) : (!emitc.lvalue<f32>) -> f32
      %198 = "emitc.call_opaque"(%195, %197, %96, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %199 = "emitc.load"(%88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %200 = "emitc.subscript"(%4, %114) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %201 = "emitc.load"(%200) : (!emitc.lvalue<f32>) -> f32
      %202 = "emitc.call_opaque"(%199, %201, %100, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %203 = "emitc.load"(%89) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %204 = "emitc.subscript"(%4, %114) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %205 = "emitc.load"(%204) : (!emitc.lvalue<f32>) -> f32
      %206 = "emitc.call_opaque"(%203, %205, %104, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %207 = "emitc.load"(%90) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %208 = "emitc.subscript"(%4, %114) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %209 = "emitc.load"(%208) : (!emitc.lvalue<f32>) -> f32
      %210 = "emitc.call_opaque"(%207, %209, %108, %93) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%67, %118) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%68, %122) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%69, %126) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%70, %130) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%71, %134) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%72, %138) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%73, %142) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%74, %146) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%75, %150) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%76, %154) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%77, %158) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%78, %162) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%79, %166) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%80, %170) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%81, %174) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%82, %178) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%83, %182) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%84, %186) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%85, %190) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%86, %194) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%87, %198) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%88, %202) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%89, %206) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%90, %210) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %211 = "emitc.load"(%67) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %212 = "emitc.load"(%68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %213 = "emitc.load"(%69) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %214 = "emitc.load"(%70) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %215 = "emitc.load"(%71) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %216 = "emitc.load"(%72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %217 = "emitc.load"(%73) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %218 = "emitc.load"(%74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %219 = "emitc.load"(%75) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %220 = "emitc.load"(%76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %221 = "emitc.load"(%77) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %222 = "emitc.load"(%78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %223 = "emitc.load"(%79) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %224 = "emitc.load"(%80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %225 = "emitc.load"(%81) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %226 = "emitc.load"(%82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %227 = "emitc.load"(%83) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %228 = "emitc.load"(%84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %229 = "emitc.load"(%85) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %230 = "emitc.load"(%86) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %231 = "emitc.load"(%87) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %232 = "emitc.load"(%88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %233 = "emitc.load"(%89) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %234 = "emitc.load"(%90) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %235 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %236 = "emitc.apply"(%235) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%236, %211, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %237 = "emitc.add"(%10, %18) : (index, index) -> index
    %238 = "emitc.subscript"(%7, %237) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %239 = "emitc.apply"(%238) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%239, %212, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %240 = "emitc.add"(%10, %26) : (index, index) -> index
    %241 = "emitc.subscript"(%7, %240) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %242 = "emitc.apply"(%241) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%242, %213, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %243 = "emitc.add"(%10, %34) : (index, index) -> index
    %244 = "emitc.subscript"(%7, %243) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %245 = "emitc.apply"(%244) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%245, %214, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %246 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %247 = "emitc.apply"(%246) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%247, %215, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %248 = "emitc.add"(%8, %18) : (index, index) -> index
    %249 = "emitc.subscript"(%7, %248) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %250 = "emitc.apply"(%249) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%250, %216, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %251 = "emitc.add"(%8, %26) : (index, index) -> index
    %252 = "emitc.subscript"(%7, %251) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %253 = "emitc.apply"(%252) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%253, %217, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %254 = "emitc.add"(%8, %34) : (index, index) -> index
    %255 = "emitc.subscript"(%7, %254) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %256 = "emitc.apply"(%255) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%256, %218, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %257 = "emitc.mul"(%12, %8) : (index, index) -> index
    %258 = "emitc.subscript"(%7, %257) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %259 = "emitc.apply"(%258) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%259, %219, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %260 = "emitc.add"(%257, %18) : (index, index) -> index
    %261 = "emitc.subscript"(%7, %260) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %262 = "emitc.apply"(%261) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%262, %220, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %263 = "emitc.add"(%257, %26) : (index, index) -> index
    %264 = "emitc.subscript"(%7, %263) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %265 = "emitc.apply"(%264) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%265, %221, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %266 = "emitc.add"(%257, %34) : (index, index) -> index
    %267 = "emitc.subscript"(%7, %266) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %268 = "emitc.apply"(%267) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%268, %222, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %269 = "emitc.mul"(%13, %8) : (index, index) -> index
    %270 = "emitc.subscript"(%7, %269) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %271 = "emitc.apply"(%270) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%271, %223, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %272 = "emitc.add"(%269, %18) : (index, index) -> index
    %273 = "emitc.subscript"(%7, %272) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %274 = "emitc.apply"(%273) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%274, %224, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %275 = "emitc.add"(%269, %26) : (index, index) -> index
    %276 = "emitc.subscript"(%7, %275) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %277 = "emitc.apply"(%276) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%277, %225, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %278 = "emitc.add"(%269, %34) : (index, index) -> index
    %279 = "emitc.subscript"(%7, %278) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %280 = "emitc.apply"(%279) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%280, %226, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %281 = "emitc.mul"(%14, %8) : (index, index) -> index
    %282 = "emitc.subscript"(%7, %281) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %283 = "emitc.apply"(%282) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%283, %227, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %284 = "emitc.add"(%281, %18) : (index, index) -> index
    %285 = "emitc.subscript"(%7, %284) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %286 = "emitc.apply"(%285) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%286, %228, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %287 = "emitc.add"(%281, %26) : (index, index) -> index
    %288 = "emitc.subscript"(%7, %287) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %289 = "emitc.apply"(%288) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%289, %229, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %290 = "emitc.add"(%281, %34) : (index, index) -> index
    %291 = "emitc.subscript"(%7, %290) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %292 = "emitc.apply"(%291) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%292, %230, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %293 = "emitc.mul"(%15, %8) : (index, index) -> index
    %294 = "emitc.subscript"(%7, %293) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %295 = "emitc.apply"(%294) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%295, %231, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %296 = "emitc.add"(%293, %18) : (index, index) -> index
    %297 = "emitc.subscript"(%7, %296) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %298 = "emitc.apply"(%297) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%298, %232, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %299 = "emitc.add"(%293, %26) : (index, index) -> index
    %300 = "emitc.subscript"(%7, %299) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %301 = "emitc.apply"(%300) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%301, %233, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %302 = "emitc.add"(%293, %34) : (index, index) -> index
    %303 = "emitc.subscript"(%7, %302) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %304 = "emitc.apply"(%303) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%304, %234, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
}) : () -> ()
