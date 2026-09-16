"builtin.module"() ({
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_16x15_b0", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
    %27 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %28 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %29 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %30 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %31 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %32 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %33 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %34 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %35 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %36 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %38 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %39 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %40 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    %42 = "emitc.call_opaque"(%9, %18) <{callee = "__riscv_vfmv_v_f_f32m1"}> : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
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
    %57 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%57, %27) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %58 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%58, %28) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %59 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%59, %29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %60 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%60, %30) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %61 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%61, %31) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %62 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%62, %32) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %63 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%63, %33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %64 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%64, %34) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %65 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%65, %35) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %66 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%66, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %67 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%67, %37) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %68 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%68, %38) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %69 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%69, %39) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %70 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%70, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %71 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%71, %41) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %72 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%72, %42) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %73 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%73, %43) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %74 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%74, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %75 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%75, %45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %76 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%76, %46) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %77 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%77, %47) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %78 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%78, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %79 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%79, %49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %80 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%80, %50) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %81 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%81, %51) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %82 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%82, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %83 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%83, %53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %84 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%84, %54) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %85 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%85, %55) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %86 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%86, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%87 : index):
      %88 = "emitc.mul"(%87, %3) : (index, index) -> index
      %89 = "emitc.constant"() {value = 8 : index} : () -> index
      %90 = "emitc.subscript"(%2, %88) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %91 = "emitc.apply"(%90) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %92 = "emitc.call_opaque"(%91, %89) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %93 = "emitc.add"(%88, %18) : (index, index) -> index
      %94 = "emitc.subscript"(%2, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %96 = "emitc.call_opaque"(%95, %89) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %97 = "emitc.mul"(%87, %5) : (index, index) -> index
      %98 = "emitc.add"(%97, %11) : (index, index) -> index
      %99 = "emitc.add"(%97, %12) : (index, index) -> index
      %100 = "emitc.add"(%97, %13) : (index, index) -> index
      %101 = "emitc.add"(%97, %14) : (index, index) -> index
      %102 = "emitc.add"(%97, %15) : (index, index) -> index
      %103 = "emitc.add"(%97, %16) : (index, index) -> index
      %104 = "emitc.add"(%97, %17) : (index, index) -> index
      %105 = "emitc.add"(%97, %18) : (index, index) -> index
      %106 = "emitc.add"(%97, %19) : (index, index) -> index
      %107 = "emitc.add"(%97, %20) : (index, index) -> index
      %108 = "emitc.add"(%97, %21) : (index, index) -> index
      %109 = "emitc.add"(%97, %22) : (index, index) -> index
      %110 = "emitc.add"(%97, %23) : (index, index) -> index
      %111 = "emitc.add"(%97, %24) : (index, index) -> index
      %112 = "emitc.load"(%57) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %113 = "emitc.subscript"(%4, %97) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %114 = "emitc.load"(%113) : (!emitc.lvalue<f32>) -> f32
      %115 = "emitc.call_opaque"(%112, %114, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %116 = "emitc.load"(%58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %117 = "emitc.subscript"(%4, %97) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %118 = "emitc.load"(%117) : (!emitc.lvalue<f32>) -> f32
      %119 = "emitc.call_opaque"(%116, %118, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %120 = "emitc.load"(%59) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %121 = "emitc.subscript"(%4, %98) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %122 = "emitc.load"(%121) : (!emitc.lvalue<f32>) -> f32
      %123 = "emitc.call_opaque"(%120, %122, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %124 = "emitc.load"(%60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %125 = "emitc.subscript"(%4, %98) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %126 = "emitc.load"(%125) : (!emitc.lvalue<f32>) -> f32
      %127 = "emitc.call_opaque"(%124, %126, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %128 = "emitc.load"(%61) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %129 = "emitc.subscript"(%4, %99) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %130 = "emitc.load"(%129) : (!emitc.lvalue<f32>) -> f32
      %131 = "emitc.call_opaque"(%128, %130, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %132 = "emitc.load"(%62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %133 = "emitc.subscript"(%4, %99) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %134 = "emitc.load"(%133) : (!emitc.lvalue<f32>) -> f32
      %135 = "emitc.call_opaque"(%132, %134, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %136 = "emitc.load"(%63) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %137 = "emitc.subscript"(%4, %100) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %138 = "emitc.load"(%137) : (!emitc.lvalue<f32>) -> f32
      %139 = "emitc.call_opaque"(%136, %138, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %140 = "emitc.load"(%64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %141 = "emitc.subscript"(%4, %100) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %142 = "emitc.load"(%141) : (!emitc.lvalue<f32>) -> f32
      %143 = "emitc.call_opaque"(%140, %142, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %144 = "emitc.load"(%65) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %145 = "emitc.subscript"(%4, %101) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %146 = "emitc.load"(%145) : (!emitc.lvalue<f32>) -> f32
      %147 = "emitc.call_opaque"(%144, %146, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %148 = "emitc.load"(%66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %149 = "emitc.subscript"(%4, %101) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %150 = "emitc.load"(%149) : (!emitc.lvalue<f32>) -> f32
      %151 = "emitc.call_opaque"(%148, %150, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %152 = "emitc.load"(%67) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %153 = "emitc.subscript"(%4, %102) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %154 = "emitc.load"(%153) : (!emitc.lvalue<f32>) -> f32
      %155 = "emitc.call_opaque"(%152, %154, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %156 = "emitc.load"(%68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %157 = "emitc.subscript"(%4, %102) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %158 = "emitc.load"(%157) : (!emitc.lvalue<f32>) -> f32
      %159 = "emitc.call_opaque"(%156, %158, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %160 = "emitc.load"(%69) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %161 = "emitc.subscript"(%4, %103) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %162 = "emitc.load"(%161) : (!emitc.lvalue<f32>) -> f32
      %163 = "emitc.call_opaque"(%160, %162, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %164 = "emitc.load"(%70) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %165 = "emitc.subscript"(%4, %103) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %166 = "emitc.load"(%165) : (!emitc.lvalue<f32>) -> f32
      %167 = "emitc.call_opaque"(%164, %166, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %168 = "emitc.load"(%71) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %169 = "emitc.subscript"(%4, %104) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %170 = "emitc.load"(%169) : (!emitc.lvalue<f32>) -> f32
      %171 = "emitc.call_opaque"(%168, %170, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %172 = "emitc.load"(%72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %173 = "emitc.subscript"(%4, %104) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %174 = "emitc.load"(%173) : (!emitc.lvalue<f32>) -> f32
      %175 = "emitc.call_opaque"(%172, %174, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %176 = "emitc.load"(%73) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %177 = "emitc.subscript"(%4, %105) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %178 = "emitc.load"(%177) : (!emitc.lvalue<f32>) -> f32
      %179 = "emitc.call_opaque"(%176, %178, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %180 = "emitc.load"(%74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %181 = "emitc.subscript"(%4, %105) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %182 = "emitc.load"(%181) : (!emitc.lvalue<f32>) -> f32
      %183 = "emitc.call_opaque"(%180, %182, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %184 = "emitc.load"(%75) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %185 = "emitc.subscript"(%4, %106) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %186 = "emitc.load"(%185) : (!emitc.lvalue<f32>) -> f32
      %187 = "emitc.call_opaque"(%184, %186, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %188 = "emitc.load"(%76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %189 = "emitc.subscript"(%4, %106) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %190 = "emitc.load"(%189) : (!emitc.lvalue<f32>) -> f32
      %191 = "emitc.call_opaque"(%188, %190, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %192 = "emitc.load"(%77) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %193 = "emitc.subscript"(%4, %107) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %194 = "emitc.load"(%193) : (!emitc.lvalue<f32>) -> f32
      %195 = "emitc.call_opaque"(%192, %194, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %196 = "emitc.load"(%78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %197 = "emitc.subscript"(%4, %107) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %198 = "emitc.load"(%197) : (!emitc.lvalue<f32>) -> f32
      %199 = "emitc.call_opaque"(%196, %198, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %200 = "emitc.load"(%79) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %201 = "emitc.subscript"(%4, %108) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %202 = "emitc.load"(%201) : (!emitc.lvalue<f32>) -> f32
      %203 = "emitc.call_opaque"(%200, %202, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %204 = "emitc.load"(%80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %205 = "emitc.subscript"(%4, %108) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %206 = "emitc.load"(%205) : (!emitc.lvalue<f32>) -> f32
      %207 = "emitc.call_opaque"(%204, %206, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %208 = "emitc.load"(%81) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %209 = "emitc.subscript"(%4, %109) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %210 = "emitc.load"(%209) : (!emitc.lvalue<f32>) -> f32
      %211 = "emitc.call_opaque"(%208, %210, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %212 = "emitc.load"(%82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %213 = "emitc.subscript"(%4, %109) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %214 = "emitc.load"(%213) : (!emitc.lvalue<f32>) -> f32
      %215 = "emitc.call_opaque"(%212, %214, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %216 = "emitc.load"(%83) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %217 = "emitc.subscript"(%4, %110) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %218 = "emitc.load"(%217) : (!emitc.lvalue<f32>) -> f32
      %219 = "emitc.call_opaque"(%216, %218, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %220 = "emitc.load"(%84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %221 = "emitc.subscript"(%4, %110) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %222 = "emitc.load"(%221) : (!emitc.lvalue<f32>) -> f32
      %223 = "emitc.call_opaque"(%220, %222, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %224 = "emitc.load"(%85) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %225 = "emitc.subscript"(%4, %111) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %226 = "emitc.load"(%225) : (!emitc.lvalue<f32>) -> f32
      %227 = "emitc.call_opaque"(%224, %226, %92, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %228 = "emitc.load"(%86) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %229 = "emitc.subscript"(%4, %111) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %230 = "emitc.load"(%229) : (!emitc.lvalue<f32>) -> f32
      %231 = "emitc.call_opaque"(%228, %230, %96, %89) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%57, %115) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%58, %119) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%59, %123) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%60, %127) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%61, %131) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%62, %135) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%63, %139) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%64, %143) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%65, %147) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%66, %151) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%67, %155) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%68, %159) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%69, %163) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%70, %167) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%71, %171) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%72, %175) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%73, %179) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%74, %183) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%75, %187) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%76, %191) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%77, %195) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%78, %199) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%79, %203) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%80, %207) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%81, %211) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%82, %215) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%83, %219) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%84, %223) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%85, %227) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%86, %231) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %232 = "emitc.load"(%57) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %233 = "emitc.load"(%58) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %234 = "emitc.load"(%59) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %235 = "emitc.load"(%60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %236 = "emitc.load"(%61) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %237 = "emitc.load"(%62) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %238 = "emitc.load"(%63) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %239 = "emitc.load"(%64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %240 = "emitc.load"(%65) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %241 = "emitc.load"(%66) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %242 = "emitc.load"(%67) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %243 = "emitc.load"(%68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %244 = "emitc.load"(%69) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %245 = "emitc.load"(%70) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %246 = "emitc.load"(%71) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %247 = "emitc.load"(%72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %248 = "emitc.load"(%73) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %249 = "emitc.load"(%74) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %250 = "emitc.load"(%75) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %251 = "emitc.load"(%76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %252 = "emitc.load"(%77) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %253 = "emitc.load"(%78) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %254 = "emitc.load"(%79) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %255 = "emitc.load"(%80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %256 = "emitc.load"(%81) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %257 = "emitc.load"(%82) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %258 = "emitc.load"(%83) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %259 = "emitc.load"(%84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %260 = "emitc.load"(%85) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %261 = "emitc.load"(%86) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %262 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %263 = "emitc.apply"(%262) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%263, %232, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %264 = "emitc.add"(%10, %18) : (index, index) -> index
    %265 = "emitc.subscript"(%7, %264) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %266 = "emitc.apply"(%265) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%266, %233, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %267 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %268 = "emitc.apply"(%267) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%268, %234, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %269 = "emitc.add"(%8, %18) : (index, index) -> index
    %270 = "emitc.subscript"(%7, %269) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %271 = "emitc.apply"(%270) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%271, %235, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %272 = "emitc.mul"(%12, %8) : (index, index) -> index
    %273 = "emitc.subscript"(%7, %272) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %274 = "emitc.apply"(%273) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%274, %236, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %275 = "emitc.add"(%272, %18) : (index, index) -> index
    %276 = "emitc.subscript"(%7, %275) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %277 = "emitc.apply"(%276) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%277, %237, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %278 = "emitc.mul"(%13, %8) : (index, index) -> index
    %279 = "emitc.subscript"(%7, %278) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %280 = "emitc.apply"(%279) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%280, %238, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %281 = "emitc.add"(%278, %18) : (index, index) -> index
    %282 = "emitc.subscript"(%7, %281) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %283 = "emitc.apply"(%282) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%283, %239, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %284 = "emitc.mul"(%14, %8) : (index, index) -> index
    %285 = "emitc.subscript"(%7, %284) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %286 = "emitc.apply"(%285) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%286, %240, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %287 = "emitc.add"(%284, %18) : (index, index) -> index
    %288 = "emitc.subscript"(%7, %287) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %289 = "emitc.apply"(%288) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%289, %241, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %290 = "emitc.mul"(%15, %8) : (index, index) -> index
    %291 = "emitc.subscript"(%7, %290) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %292 = "emitc.apply"(%291) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%292, %242, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %293 = "emitc.add"(%290, %18) : (index, index) -> index
    %294 = "emitc.subscript"(%7, %293) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %295 = "emitc.apply"(%294) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%295, %243, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %296 = "emitc.mul"(%16, %8) : (index, index) -> index
    %297 = "emitc.subscript"(%7, %296) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %298 = "emitc.apply"(%297) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%298, %244, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %299 = "emitc.add"(%296, %18) : (index, index) -> index
    %300 = "emitc.subscript"(%7, %299) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %301 = "emitc.apply"(%300) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%301, %245, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %302 = "emitc.mul"(%17, %8) : (index, index) -> index
    %303 = "emitc.subscript"(%7, %302) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %304 = "emitc.apply"(%303) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%304, %246, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %305 = "emitc.add"(%302, %18) : (index, index) -> index
    %306 = "emitc.subscript"(%7, %305) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %307 = "emitc.apply"(%306) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%307, %247, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %308 = "emitc.mul"(%18, %8) : (index, index) -> index
    %309 = "emitc.subscript"(%7, %308) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %310 = "emitc.apply"(%309) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%310, %248, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %311 = "emitc.add"(%308, %18) : (index, index) -> index
    %312 = "emitc.subscript"(%7, %311) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %313 = "emitc.apply"(%312) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%313, %249, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %314 = "emitc.mul"(%19, %8) : (index, index) -> index
    %315 = "emitc.subscript"(%7, %314) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %316 = "emitc.apply"(%315) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%316, %250, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %317 = "emitc.add"(%314, %18) : (index, index) -> index
    %318 = "emitc.subscript"(%7, %317) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %319 = "emitc.apply"(%318) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%319, %251, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %320 = "emitc.mul"(%20, %8) : (index, index) -> index
    %321 = "emitc.subscript"(%7, %320) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %322 = "emitc.apply"(%321) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%322, %252, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %323 = "emitc.add"(%320, %18) : (index, index) -> index
    %324 = "emitc.subscript"(%7, %323) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %325 = "emitc.apply"(%324) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%325, %253, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %326 = "emitc.mul"(%21, %8) : (index, index) -> index
    %327 = "emitc.subscript"(%7, %326) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %328 = "emitc.apply"(%327) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%328, %254, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %329 = "emitc.add"(%326, %18) : (index, index) -> index
    %330 = "emitc.subscript"(%7, %329) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %331 = "emitc.apply"(%330) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%331, %255, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %332 = "emitc.mul"(%22, %8) : (index, index) -> index
    %333 = "emitc.subscript"(%7, %332) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %334 = "emitc.apply"(%333) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%334, %256, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %335 = "emitc.add"(%332, %18) : (index, index) -> index
    %336 = "emitc.subscript"(%7, %335) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %337 = "emitc.apply"(%336) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%337, %257, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %338 = "emitc.mul"(%23, %8) : (index, index) -> index
    %339 = "emitc.subscript"(%7, %338) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %340 = "emitc.apply"(%339) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%340, %258, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %341 = "emitc.add"(%338, %18) : (index, index) -> index
    %342 = "emitc.subscript"(%7, %341) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %343 = "emitc.apply"(%342) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%343, %259, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %344 = "emitc.mul"(%24, %8) : (index, index) -> index
    %345 = "emitc.subscript"(%7, %344) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %346 = "emitc.apply"(%345) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%346, %260, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %347 = "emitc.add"(%344, %18) : (index, index) -> index
    %348 = "emitc.subscript"(%7, %347) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %349 = "emitc.apply"(%348) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%349, %261, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
}) : () -> ()
