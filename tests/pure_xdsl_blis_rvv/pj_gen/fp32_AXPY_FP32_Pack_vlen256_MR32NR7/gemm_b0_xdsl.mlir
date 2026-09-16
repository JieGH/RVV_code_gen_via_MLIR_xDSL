"builtin.module"() ({
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_32x7_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
    %67 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %68 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %69 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %70 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %71 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%71, %43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %72 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%72, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %73 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%73, %45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %74 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%74, %46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %75 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%75, %47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %76 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%76, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %77 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%77, %49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %78 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%78, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %79 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%79, %51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %80 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%80, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %81 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%81, %53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %82 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%82, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %83 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%83, %55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %84 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%84, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %85 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%85, %57) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %86 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%86, %58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %87 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%87, %59) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %88 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%88, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %89 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%89, %61) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %90 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%90, %62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %91 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%91, %63) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %92 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%92, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %93 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%93, %65) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %94 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%94, %66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %95 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%95, %67) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %96 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%96, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %97 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%97, %69) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %98 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%98, %70) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%99 : index):
      %100 = "emitc.mul"(%99, %3) : (index, index) -> index
      %101 = "emitc.constant"() {value = 8 : index} : () -> index
      %102 = "emitc.subscript"(%2, %100) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %103 = "emitc.apply"(%102) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %104 = "emitc.call_opaque"(%103, %101) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %105 = "emitc.add"(%100, %18) : (index, index) -> index
      %106 = "emitc.subscript"(%2, %105) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %107 = "emitc.apply"(%106) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %108 = "emitc.call_opaque"(%107, %101) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %109 = "emitc.add"(%100, %26) : (index, index) -> index
      %110 = "emitc.subscript"(%2, %109) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %111 = "emitc.apply"(%110) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %112 = "emitc.call_opaque"(%111, %101) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %113 = "emitc.add"(%100, %34) : (index, index) -> index
      %114 = "emitc.subscript"(%2, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %115 = "emitc.apply"(%114) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %116 = "emitc.call_opaque"(%115, %101) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %117 = "emitc.mul"(%99, %5) : (index, index) -> index
      %118 = "emitc.add"(%117, %11) : (index, index) -> index
      %119 = "emitc.add"(%117, %12) : (index, index) -> index
      %120 = "emitc.add"(%117, %13) : (index, index) -> index
      %121 = "emitc.add"(%117, %14) : (index, index) -> index
      %122 = "emitc.add"(%117, %15) : (index, index) -> index
      %123 = "emitc.add"(%117, %16) : (index, index) -> index
      %124 = "emitc.load"(%71) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %125 = "emitc.subscript"(%4, %117) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %126 = "emitc.load"(%125) : (!emitc.lvalue<f32>) -> f32
      %127 = "emitc.call_opaque"(%124, %126, %104, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %128 = "emitc.load"(%72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %129 = "emitc.subscript"(%4, %117) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %130 = "emitc.load"(%129) : (!emitc.lvalue<f32>) -> f32
      %131 = "emitc.call_opaque"(%128, %130, %108, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %132 = "emitc.load"(%73) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %133 = "emitc.subscript"(%4, %117) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %134 = "emitc.load"(%133) : (!emitc.lvalue<f32>) -> f32
      %135 = "emitc.call_opaque"(%132, %134, %112, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %136 = "emitc.load"(%74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %137 = "emitc.subscript"(%4, %117) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %138 = "emitc.load"(%137) : (!emitc.lvalue<f32>) -> f32
      %139 = "emitc.call_opaque"(%136, %138, %116, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %140 = "emitc.load"(%75) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %141 = "emitc.subscript"(%4, %118) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %142 = "emitc.load"(%141) : (!emitc.lvalue<f32>) -> f32
      %143 = "emitc.call_opaque"(%140, %142, %104, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %144 = "emitc.load"(%76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %145 = "emitc.subscript"(%4, %118) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %146 = "emitc.load"(%145) : (!emitc.lvalue<f32>) -> f32
      %147 = "emitc.call_opaque"(%144, %146, %108, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %148 = "emitc.load"(%77) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %149 = "emitc.subscript"(%4, %118) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %150 = "emitc.load"(%149) : (!emitc.lvalue<f32>) -> f32
      %151 = "emitc.call_opaque"(%148, %150, %112, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %152 = "emitc.load"(%78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %153 = "emitc.subscript"(%4, %118) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %154 = "emitc.load"(%153) : (!emitc.lvalue<f32>) -> f32
      %155 = "emitc.call_opaque"(%152, %154, %116, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %156 = "emitc.load"(%79) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %157 = "emitc.subscript"(%4, %119) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %158 = "emitc.load"(%157) : (!emitc.lvalue<f32>) -> f32
      %159 = "emitc.call_opaque"(%156, %158, %104, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %160 = "emitc.load"(%80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %161 = "emitc.subscript"(%4, %119) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %162 = "emitc.load"(%161) : (!emitc.lvalue<f32>) -> f32
      %163 = "emitc.call_opaque"(%160, %162, %108, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %164 = "emitc.load"(%81) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %165 = "emitc.subscript"(%4, %119) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %166 = "emitc.load"(%165) : (!emitc.lvalue<f32>) -> f32
      %167 = "emitc.call_opaque"(%164, %166, %112, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %168 = "emitc.load"(%82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %169 = "emitc.subscript"(%4, %119) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %170 = "emitc.load"(%169) : (!emitc.lvalue<f32>) -> f32
      %171 = "emitc.call_opaque"(%168, %170, %116, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %172 = "emitc.load"(%83) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %173 = "emitc.subscript"(%4, %120) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %174 = "emitc.load"(%173) : (!emitc.lvalue<f32>) -> f32
      %175 = "emitc.call_opaque"(%172, %174, %104, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %176 = "emitc.load"(%84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %177 = "emitc.subscript"(%4, %120) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %178 = "emitc.load"(%177) : (!emitc.lvalue<f32>) -> f32
      %179 = "emitc.call_opaque"(%176, %178, %108, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %180 = "emitc.load"(%85) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %181 = "emitc.subscript"(%4, %120) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %182 = "emitc.load"(%181) : (!emitc.lvalue<f32>) -> f32
      %183 = "emitc.call_opaque"(%180, %182, %112, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %184 = "emitc.load"(%86) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %185 = "emitc.subscript"(%4, %120) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %186 = "emitc.load"(%185) : (!emitc.lvalue<f32>) -> f32
      %187 = "emitc.call_opaque"(%184, %186, %116, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %188 = "emitc.load"(%87) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %189 = "emitc.subscript"(%4, %121) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %190 = "emitc.load"(%189) : (!emitc.lvalue<f32>) -> f32
      %191 = "emitc.call_opaque"(%188, %190, %104, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %192 = "emitc.load"(%88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %193 = "emitc.subscript"(%4, %121) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %194 = "emitc.load"(%193) : (!emitc.lvalue<f32>) -> f32
      %195 = "emitc.call_opaque"(%192, %194, %108, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %196 = "emitc.load"(%89) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %197 = "emitc.subscript"(%4, %121) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %198 = "emitc.load"(%197) : (!emitc.lvalue<f32>) -> f32
      %199 = "emitc.call_opaque"(%196, %198, %112, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %200 = "emitc.load"(%90) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %201 = "emitc.subscript"(%4, %121) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %202 = "emitc.load"(%201) : (!emitc.lvalue<f32>) -> f32
      %203 = "emitc.call_opaque"(%200, %202, %116, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %204 = "emitc.load"(%91) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %205 = "emitc.subscript"(%4, %122) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %206 = "emitc.load"(%205) : (!emitc.lvalue<f32>) -> f32
      %207 = "emitc.call_opaque"(%204, %206, %104, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %208 = "emitc.load"(%92) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %209 = "emitc.subscript"(%4, %122) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %210 = "emitc.load"(%209) : (!emitc.lvalue<f32>) -> f32
      %211 = "emitc.call_opaque"(%208, %210, %108, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %212 = "emitc.load"(%93) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %213 = "emitc.subscript"(%4, %122) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %214 = "emitc.load"(%213) : (!emitc.lvalue<f32>) -> f32
      %215 = "emitc.call_opaque"(%212, %214, %112, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %216 = "emitc.load"(%94) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %217 = "emitc.subscript"(%4, %122) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %218 = "emitc.load"(%217) : (!emitc.lvalue<f32>) -> f32
      %219 = "emitc.call_opaque"(%216, %218, %116, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %220 = "emitc.load"(%95) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %221 = "emitc.subscript"(%4, %123) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %222 = "emitc.load"(%221) : (!emitc.lvalue<f32>) -> f32
      %223 = "emitc.call_opaque"(%220, %222, %104, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %224 = "emitc.load"(%96) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %225 = "emitc.subscript"(%4, %123) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %226 = "emitc.load"(%225) : (!emitc.lvalue<f32>) -> f32
      %227 = "emitc.call_opaque"(%224, %226, %108, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %228 = "emitc.load"(%97) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %229 = "emitc.subscript"(%4, %123) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %230 = "emitc.load"(%229) : (!emitc.lvalue<f32>) -> f32
      %231 = "emitc.call_opaque"(%228, %230, %112, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %232 = "emitc.load"(%98) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %233 = "emitc.subscript"(%4, %123) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %234 = "emitc.load"(%233) : (!emitc.lvalue<f32>) -> f32
      %235 = "emitc.call_opaque"(%232, %234, %116, %101) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%71, %127) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%72, %131) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%73, %135) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%74, %139) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%75, %143) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%76, %147) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%77, %151) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%78, %155) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%79, %159) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%80, %163) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%81, %167) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%82, %171) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%83, %175) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%84, %179) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%85, %183) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%86, %187) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%87, %191) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%88, %195) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%89, %199) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%90, %203) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%91, %207) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%92, %211) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%93, %215) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%94, %219) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%95, %223) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%96, %227) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%97, %231) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%98, %235) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %236 = "emitc.load"(%71) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %237 = "emitc.load"(%72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %238 = "emitc.load"(%73) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %239 = "emitc.load"(%74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %240 = "emitc.load"(%75) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %241 = "emitc.load"(%76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %242 = "emitc.load"(%77) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %243 = "emitc.load"(%78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %244 = "emitc.load"(%79) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %245 = "emitc.load"(%80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %246 = "emitc.load"(%81) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %247 = "emitc.load"(%82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %248 = "emitc.load"(%83) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %249 = "emitc.load"(%84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %250 = "emitc.load"(%85) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %251 = "emitc.load"(%86) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %252 = "emitc.load"(%87) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %253 = "emitc.load"(%88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %254 = "emitc.load"(%89) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %255 = "emitc.load"(%90) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %256 = "emitc.load"(%91) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %257 = "emitc.load"(%92) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %258 = "emitc.load"(%93) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %259 = "emitc.load"(%94) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %260 = "emitc.load"(%95) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %261 = "emitc.load"(%96) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %262 = "emitc.load"(%97) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %263 = "emitc.load"(%98) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %264 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %265 = "emitc.apply"(%264) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%265, %236, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %266 = "emitc.add"(%10, %18) : (index, index) -> index
    %267 = "emitc.subscript"(%7, %266) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %268 = "emitc.apply"(%267) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%268, %237, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %269 = "emitc.add"(%10, %26) : (index, index) -> index
    %270 = "emitc.subscript"(%7, %269) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %271 = "emitc.apply"(%270) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%271, %238, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %272 = "emitc.add"(%10, %34) : (index, index) -> index
    %273 = "emitc.subscript"(%7, %272) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %274 = "emitc.apply"(%273) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%274, %239, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %275 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %276 = "emitc.apply"(%275) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%276, %240, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %277 = "emitc.add"(%8, %18) : (index, index) -> index
    %278 = "emitc.subscript"(%7, %277) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %279 = "emitc.apply"(%278) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%279, %241, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %280 = "emitc.add"(%8, %26) : (index, index) -> index
    %281 = "emitc.subscript"(%7, %280) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %282 = "emitc.apply"(%281) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%282, %242, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %283 = "emitc.add"(%8, %34) : (index, index) -> index
    %284 = "emitc.subscript"(%7, %283) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %285 = "emitc.apply"(%284) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%285, %243, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %286 = "emitc.mul"(%12, %8) : (index, index) -> index
    %287 = "emitc.subscript"(%7, %286) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %288 = "emitc.apply"(%287) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%288, %244, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %289 = "emitc.add"(%286, %18) : (index, index) -> index
    %290 = "emitc.subscript"(%7, %289) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %291 = "emitc.apply"(%290) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%291, %245, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %292 = "emitc.add"(%286, %26) : (index, index) -> index
    %293 = "emitc.subscript"(%7, %292) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %294 = "emitc.apply"(%293) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%294, %246, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %295 = "emitc.add"(%286, %34) : (index, index) -> index
    %296 = "emitc.subscript"(%7, %295) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %297 = "emitc.apply"(%296) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%297, %247, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %298 = "emitc.mul"(%13, %8) : (index, index) -> index
    %299 = "emitc.subscript"(%7, %298) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %300 = "emitc.apply"(%299) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%300, %248, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %301 = "emitc.add"(%298, %18) : (index, index) -> index
    %302 = "emitc.subscript"(%7, %301) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %303 = "emitc.apply"(%302) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%303, %249, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %304 = "emitc.add"(%298, %26) : (index, index) -> index
    %305 = "emitc.subscript"(%7, %304) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %306 = "emitc.apply"(%305) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%306, %250, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %307 = "emitc.add"(%298, %34) : (index, index) -> index
    %308 = "emitc.subscript"(%7, %307) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %309 = "emitc.apply"(%308) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%309, %251, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %310 = "emitc.mul"(%14, %8) : (index, index) -> index
    %311 = "emitc.subscript"(%7, %310) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %312 = "emitc.apply"(%311) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%312, %252, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %313 = "emitc.add"(%310, %18) : (index, index) -> index
    %314 = "emitc.subscript"(%7, %313) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %315 = "emitc.apply"(%314) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%315, %253, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %316 = "emitc.add"(%310, %26) : (index, index) -> index
    %317 = "emitc.subscript"(%7, %316) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %318 = "emitc.apply"(%317) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%318, %254, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %319 = "emitc.add"(%310, %34) : (index, index) -> index
    %320 = "emitc.subscript"(%7, %319) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %321 = "emitc.apply"(%320) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%321, %255, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %322 = "emitc.mul"(%15, %8) : (index, index) -> index
    %323 = "emitc.subscript"(%7, %322) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %324 = "emitc.apply"(%323) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%324, %256, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %325 = "emitc.add"(%322, %18) : (index, index) -> index
    %326 = "emitc.subscript"(%7, %325) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %327 = "emitc.apply"(%326) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%327, %257, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %328 = "emitc.add"(%322, %26) : (index, index) -> index
    %329 = "emitc.subscript"(%7, %328) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %330 = "emitc.apply"(%329) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%330, %258, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %331 = "emitc.add"(%322, %34) : (index, index) -> index
    %332 = "emitc.subscript"(%7, %331) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %333 = "emitc.apply"(%332) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%333, %259, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %334 = "emitc.mul"(%16, %8) : (index, index) -> index
    %335 = "emitc.subscript"(%7, %334) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %336 = "emitc.apply"(%335) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%336, %260, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %337 = "emitc.add"(%334, %18) : (index, index) -> index
    %338 = "emitc.subscript"(%7, %337) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %339 = "emitc.apply"(%338) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%339, %261, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %340 = "emitc.add"(%334, %26) : (index, index) -> index
    %341 = "emitc.subscript"(%7, %340) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %342 = "emitc.apply"(%341) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%342, %262, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %343 = "emitc.add"(%334, %34) : (index, index) -> index
    %344 = "emitc.subscript"(%7, %343) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %345 = "emitc.apply"(%344) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%345, %263, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
}) : () -> ()
