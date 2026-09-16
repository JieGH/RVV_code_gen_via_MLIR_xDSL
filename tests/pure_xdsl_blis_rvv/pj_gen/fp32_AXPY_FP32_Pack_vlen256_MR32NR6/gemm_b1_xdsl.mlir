"builtin.module"() ({
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_32x6_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
    %43 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %44 = "emitc.apply"(%43) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %45 = "emitc.call_opaque"(%44, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %46 = "emitc.add"(%10, %18) : (index, index) -> index
    %47 = "emitc.subscript"(%7, %46) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %48 = "emitc.apply"(%47) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %49 = "emitc.call_opaque"(%48, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %50 = "emitc.add"(%10, %26) : (index, index) -> index
    %51 = "emitc.subscript"(%7, %50) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %52 = "emitc.apply"(%51) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %53 = "emitc.call_opaque"(%52, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %54 = "emitc.add"(%10, %34) : (index, index) -> index
    %55 = "emitc.subscript"(%7, %54) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %56 = "emitc.apply"(%55) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %57 = "emitc.call_opaque"(%56, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %58 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %59 = "emitc.apply"(%58) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %60 = "emitc.call_opaque"(%59, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %61 = "emitc.add"(%8, %18) : (index, index) -> index
    %62 = "emitc.subscript"(%7, %61) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %63 = "emitc.apply"(%62) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %64 = "emitc.call_opaque"(%63, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.add"(%8, %26) : (index, index) -> index
    %66 = "emitc.subscript"(%7, %65) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %67 = "emitc.apply"(%66) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %68 = "emitc.call_opaque"(%67, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %69 = "emitc.add"(%8, %34) : (index, index) -> index
    %70 = "emitc.subscript"(%7, %69) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %71 = "emitc.apply"(%70) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %72 = "emitc.call_opaque"(%71, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %73 = "emitc.mul"(%12, %8) : (index, index) -> index
    %74 = "emitc.subscript"(%7, %73) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %75 = "emitc.apply"(%74) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %76 = "emitc.call_opaque"(%75, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %77 = "emitc.add"(%73, %18) : (index, index) -> index
    %78 = "emitc.subscript"(%7, %77) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %79 = "emitc.apply"(%78) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %80 = "emitc.call_opaque"(%79, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %81 = "emitc.add"(%73, %26) : (index, index) -> index
    %82 = "emitc.subscript"(%7, %81) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %83 = "emitc.apply"(%82) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %84 = "emitc.call_opaque"(%83, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %85 = "emitc.add"(%73, %34) : (index, index) -> index
    %86 = "emitc.subscript"(%7, %85) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %87 = "emitc.apply"(%86) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %88 = "emitc.call_opaque"(%87, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %89 = "emitc.mul"(%13, %8) : (index, index) -> index
    %90 = "emitc.subscript"(%7, %89) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %91 = "emitc.apply"(%90) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %92 = "emitc.call_opaque"(%91, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %93 = "emitc.add"(%89, %18) : (index, index) -> index
    %94 = "emitc.subscript"(%7, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %96 = "emitc.call_opaque"(%95, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %97 = "emitc.add"(%89, %26) : (index, index) -> index
    %98 = "emitc.subscript"(%7, %97) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %99 = "emitc.apply"(%98) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %100 = "emitc.call_opaque"(%99, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %101 = "emitc.add"(%89, %34) : (index, index) -> index
    %102 = "emitc.subscript"(%7, %101) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %103 = "emitc.apply"(%102) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %104 = "emitc.call_opaque"(%103, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %105 = "emitc.mul"(%14, %8) : (index, index) -> index
    %106 = "emitc.subscript"(%7, %105) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %107 = "emitc.apply"(%106) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %108 = "emitc.call_opaque"(%107, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %109 = "emitc.add"(%105, %18) : (index, index) -> index
    %110 = "emitc.subscript"(%7, %109) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %111 = "emitc.apply"(%110) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %112 = "emitc.call_opaque"(%111, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %113 = "emitc.add"(%105, %26) : (index, index) -> index
    %114 = "emitc.subscript"(%7, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %115 = "emitc.apply"(%114) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %116 = "emitc.call_opaque"(%115, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %117 = "emitc.add"(%105, %34) : (index, index) -> index
    %118 = "emitc.subscript"(%7, %117) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %119 = "emitc.apply"(%118) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %120 = "emitc.call_opaque"(%119, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %121 = "emitc.mul"(%15, %8) : (index, index) -> index
    %122 = "emitc.subscript"(%7, %121) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %123 = "emitc.apply"(%122) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %124 = "emitc.call_opaque"(%123, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %125 = "emitc.add"(%121, %18) : (index, index) -> index
    %126 = "emitc.subscript"(%7, %125) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %127 = "emitc.apply"(%126) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %128 = "emitc.call_opaque"(%127, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %129 = "emitc.add"(%121, %26) : (index, index) -> index
    %130 = "emitc.subscript"(%7, %129) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %131 = "emitc.apply"(%130) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %132 = "emitc.call_opaque"(%131, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %133 = "emitc.add"(%121, %34) : (index, index) -> index
    %134 = "emitc.subscript"(%7, %133) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %135 = "emitc.apply"(%134) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %136 = "emitc.call_opaque"(%135, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %137 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%137, %45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %138 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%138, %49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %139 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%139, %53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %140 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%140, %57) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %141 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%141, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %142 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%142, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %143 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%143, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %144 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%144, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %145 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%145, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %146 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%146, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %147 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%147, %84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %148 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%148, %88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %149 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%149, %92) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %150 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%150, %96) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %151 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%151, %100) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %152 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%152, %104) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %153 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%153, %108) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %154 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%154, %112) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %155 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%155, %116) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %156 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%156, %120) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %157 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%157, %124) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %158 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%158, %128) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %159 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%159, %132) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %160 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%160, %136) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%161 : index):
      %162 = "emitc.mul"(%161, %3) : (index, index) -> index
      %163 = "emitc.constant"() {value = 8 : index} : () -> index
      %164 = "emitc.subscript"(%2, %162) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %165 = "emitc.apply"(%164) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %166 = "emitc.call_opaque"(%165, %163) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %167 = "emitc.add"(%162, %18) : (index, index) -> index
      %168 = "emitc.subscript"(%2, %167) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %169 = "emitc.apply"(%168) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %170 = "emitc.call_opaque"(%169, %163) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %171 = "emitc.add"(%162, %26) : (index, index) -> index
      %172 = "emitc.subscript"(%2, %171) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %173 = "emitc.apply"(%172) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %174 = "emitc.call_opaque"(%173, %163) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %175 = "emitc.add"(%162, %34) : (index, index) -> index
      %176 = "emitc.subscript"(%2, %175) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %177 = "emitc.apply"(%176) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %178 = "emitc.call_opaque"(%177, %163) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %179 = "emitc.mul"(%161, %5) : (index, index) -> index
      %180 = "emitc.add"(%179, %11) : (index, index) -> index
      %181 = "emitc.add"(%179, %12) : (index, index) -> index
      %182 = "emitc.add"(%179, %13) : (index, index) -> index
      %183 = "emitc.add"(%179, %14) : (index, index) -> index
      %184 = "emitc.add"(%179, %15) : (index, index) -> index
      %185 = "emitc.load"(%137) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %186 = "emitc.subscript"(%4, %179) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %187 = "emitc.load"(%186) : (!emitc.lvalue<f32>) -> f32
      %188 = "emitc.call_opaque"(%185, %187, %166, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %189 = "emitc.load"(%138) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %190 = "emitc.subscript"(%4, %179) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %191 = "emitc.load"(%190) : (!emitc.lvalue<f32>) -> f32
      %192 = "emitc.call_opaque"(%189, %191, %170, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %193 = "emitc.load"(%139) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %194 = "emitc.subscript"(%4, %179) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %195 = "emitc.load"(%194) : (!emitc.lvalue<f32>) -> f32
      %196 = "emitc.call_opaque"(%193, %195, %174, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %197 = "emitc.load"(%140) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %198 = "emitc.subscript"(%4, %179) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %199 = "emitc.load"(%198) : (!emitc.lvalue<f32>) -> f32
      %200 = "emitc.call_opaque"(%197, %199, %178, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %201 = "emitc.load"(%141) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %202 = "emitc.subscript"(%4, %180) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %203 = "emitc.load"(%202) : (!emitc.lvalue<f32>) -> f32
      %204 = "emitc.call_opaque"(%201, %203, %166, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %205 = "emitc.load"(%142) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %206 = "emitc.subscript"(%4, %180) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %207 = "emitc.load"(%206) : (!emitc.lvalue<f32>) -> f32
      %208 = "emitc.call_opaque"(%205, %207, %170, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %209 = "emitc.load"(%143) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %210 = "emitc.subscript"(%4, %180) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %211 = "emitc.load"(%210) : (!emitc.lvalue<f32>) -> f32
      %212 = "emitc.call_opaque"(%209, %211, %174, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %213 = "emitc.load"(%144) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %214 = "emitc.subscript"(%4, %180) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %215 = "emitc.load"(%214) : (!emitc.lvalue<f32>) -> f32
      %216 = "emitc.call_opaque"(%213, %215, %178, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %217 = "emitc.load"(%145) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %218 = "emitc.subscript"(%4, %181) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %219 = "emitc.load"(%218) : (!emitc.lvalue<f32>) -> f32
      %220 = "emitc.call_opaque"(%217, %219, %166, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %221 = "emitc.load"(%146) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %222 = "emitc.subscript"(%4, %181) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %223 = "emitc.load"(%222) : (!emitc.lvalue<f32>) -> f32
      %224 = "emitc.call_opaque"(%221, %223, %170, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %225 = "emitc.load"(%147) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %226 = "emitc.subscript"(%4, %181) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %227 = "emitc.load"(%226) : (!emitc.lvalue<f32>) -> f32
      %228 = "emitc.call_opaque"(%225, %227, %174, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %229 = "emitc.load"(%148) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %230 = "emitc.subscript"(%4, %181) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %231 = "emitc.load"(%230) : (!emitc.lvalue<f32>) -> f32
      %232 = "emitc.call_opaque"(%229, %231, %178, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %233 = "emitc.load"(%149) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %234 = "emitc.subscript"(%4, %182) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %235 = "emitc.load"(%234) : (!emitc.lvalue<f32>) -> f32
      %236 = "emitc.call_opaque"(%233, %235, %166, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %237 = "emitc.load"(%150) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %238 = "emitc.subscript"(%4, %182) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %239 = "emitc.load"(%238) : (!emitc.lvalue<f32>) -> f32
      %240 = "emitc.call_opaque"(%237, %239, %170, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %241 = "emitc.load"(%151) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %242 = "emitc.subscript"(%4, %182) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %243 = "emitc.load"(%242) : (!emitc.lvalue<f32>) -> f32
      %244 = "emitc.call_opaque"(%241, %243, %174, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %245 = "emitc.load"(%152) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %246 = "emitc.subscript"(%4, %182) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %247 = "emitc.load"(%246) : (!emitc.lvalue<f32>) -> f32
      %248 = "emitc.call_opaque"(%245, %247, %178, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %249 = "emitc.load"(%153) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %250 = "emitc.subscript"(%4, %183) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %251 = "emitc.load"(%250) : (!emitc.lvalue<f32>) -> f32
      %252 = "emitc.call_opaque"(%249, %251, %166, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %253 = "emitc.load"(%154) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %254 = "emitc.subscript"(%4, %183) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %255 = "emitc.load"(%254) : (!emitc.lvalue<f32>) -> f32
      %256 = "emitc.call_opaque"(%253, %255, %170, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %257 = "emitc.load"(%155) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %258 = "emitc.subscript"(%4, %183) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %259 = "emitc.load"(%258) : (!emitc.lvalue<f32>) -> f32
      %260 = "emitc.call_opaque"(%257, %259, %174, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %261 = "emitc.load"(%156) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %262 = "emitc.subscript"(%4, %183) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %263 = "emitc.load"(%262) : (!emitc.lvalue<f32>) -> f32
      %264 = "emitc.call_opaque"(%261, %263, %178, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %265 = "emitc.load"(%157) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %266 = "emitc.subscript"(%4, %184) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %267 = "emitc.load"(%266) : (!emitc.lvalue<f32>) -> f32
      %268 = "emitc.call_opaque"(%265, %267, %166, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %269 = "emitc.load"(%158) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %270 = "emitc.subscript"(%4, %184) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %271 = "emitc.load"(%270) : (!emitc.lvalue<f32>) -> f32
      %272 = "emitc.call_opaque"(%269, %271, %170, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %273 = "emitc.load"(%159) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %274 = "emitc.subscript"(%4, %184) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %275 = "emitc.load"(%274) : (!emitc.lvalue<f32>) -> f32
      %276 = "emitc.call_opaque"(%273, %275, %174, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %277 = "emitc.load"(%160) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %278 = "emitc.subscript"(%4, %184) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %279 = "emitc.load"(%278) : (!emitc.lvalue<f32>) -> f32
      %280 = "emitc.call_opaque"(%277, %279, %178, %163) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%137, %188) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%138, %192) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%139, %196) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%140, %200) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%141, %204) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%142, %208) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%143, %212) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%144, %216) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%145, %220) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%146, %224) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%147, %228) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%148, %232) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%149, %236) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%150, %240) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%151, %244) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%152, %248) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%153, %252) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%154, %256) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%155, %260) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%156, %264) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%157, %268) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%158, %272) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%159, %276) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%160, %280) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %281 = "emitc.load"(%137) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %282 = "emitc.load"(%138) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %283 = "emitc.load"(%139) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %284 = "emitc.load"(%140) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %285 = "emitc.load"(%141) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %286 = "emitc.load"(%142) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %287 = "emitc.load"(%143) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %288 = "emitc.load"(%144) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %289 = "emitc.load"(%145) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %290 = "emitc.load"(%146) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %291 = "emitc.load"(%147) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %292 = "emitc.load"(%148) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %293 = "emitc.load"(%149) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %294 = "emitc.load"(%150) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %295 = "emitc.load"(%151) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %296 = "emitc.load"(%152) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %297 = "emitc.load"(%153) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %298 = "emitc.load"(%154) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %299 = "emitc.load"(%155) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %300 = "emitc.load"(%156) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %301 = "emitc.load"(%157) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %302 = "emitc.load"(%158) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %303 = "emitc.load"(%159) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %304 = "emitc.load"(%160) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %305 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %306 = "emitc.apply"(%305) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%306, %281, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %307 = "emitc.add"(%10, %18) : (index, index) -> index
    %308 = "emitc.subscript"(%7, %307) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %309 = "emitc.apply"(%308) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%309, %282, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %310 = "emitc.add"(%10, %26) : (index, index) -> index
    %311 = "emitc.subscript"(%7, %310) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %312 = "emitc.apply"(%311) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%312, %283, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %313 = "emitc.add"(%10, %34) : (index, index) -> index
    %314 = "emitc.subscript"(%7, %313) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %315 = "emitc.apply"(%314) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%315, %284, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %316 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %317 = "emitc.apply"(%316) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%317, %285, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %318 = "emitc.add"(%8, %18) : (index, index) -> index
    %319 = "emitc.subscript"(%7, %318) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %320 = "emitc.apply"(%319) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%320, %286, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %321 = "emitc.add"(%8, %26) : (index, index) -> index
    %322 = "emitc.subscript"(%7, %321) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %323 = "emitc.apply"(%322) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%323, %287, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %324 = "emitc.add"(%8, %34) : (index, index) -> index
    %325 = "emitc.subscript"(%7, %324) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %326 = "emitc.apply"(%325) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%326, %288, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %327 = "emitc.mul"(%12, %8) : (index, index) -> index
    %328 = "emitc.subscript"(%7, %327) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %329 = "emitc.apply"(%328) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%329, %289, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %330 = "emitc.add"(%327, %18) : (index, index) -> index
    %331 = "emitc.subscript"(%7, %330) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %332 = "emitc.apply"(%331) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%332, %290, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %333 = "emitc.add"(%327, %26) : (index, index) -> index
    %334 = "emitc.subscript"(%7, %333) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %335 = "emitc.apply"(%334) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%335, %291, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %336 = "emitc.add"(%327, %34) : (index, index) -> index
    %337 = "emitc.subscript"(%7, %336) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %338 = "emitc.apply"(%337) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%338, %292, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %339 = "emitc.mul"(%13, %8) : (index, index) -> index
    %340 = "emitc.subscript"(%7, %339) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %341 = "emitc.apply"(%340) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%341, %293, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %342 = "emitc.add"(%339, %18) : (index, index) -> index
    %343 = "emitc.subscript"(%7, %342) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %344 = "emitc.apply"(%343) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%344, %294, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %345 = "emitc.add"(%339, %26) : (index, index) -> index
    %346 = "emitc.subscript"(%7, %345) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %347 = "emitc.apply"(%346) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%347, %295, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %348 = "emitc.add"(%339, %34) : (index, index) -> index
    %349 = "emitc.subscript"(%7, %348) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %350 = "emitc.apply"(%349) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%350, %296, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %351 = "emitc.mul"(%14, %8) : (index, index) -> index
    %352 = "emitc.subscript"(%7, %351) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %353 = "emitc.apply"(%352) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%353, %297, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %354 = "emitc.add"(%351, %18) : (index, index) -> index
    %355 = "emitc.subscript"(%7, %354) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %356 = "emitc.apply"(%355) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%356, %298, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %357 = "emitc.add"(%351, %26) : (index, index) -> index
    %358 = "emitc.subscript"(%7, %357) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %359 = "emitc.apply"(%358) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%359, %299, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %360 = "emitc.add"(%351, %34) : (index, index) -> index
    %361 = "emitc.subscript"(%7, %360) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %362 = "emitc.apply"(%361) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%362, %300, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %363 = "emitc.mul"(%15, %8) : (index, index) -> index
    %364 = "emitc.subscript"(%7, %363) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %365 = "emitc.apply"(%364) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%365, %301, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %366 = "emitc.add"(%363, %18) : (index, index) -> index
    %367 = "emitc.subscript"(%7, %366) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %368 = "emitc.apply"(%367) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%368, %302, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %369 = "emitc.add"(%363, %26) : (index, index) -> index
    %370 = "emitc.subscript"(%7, %369) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %371 = "emitc.apply"(%370) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%371, %303, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %372 = "emitc.add"(%363, %34) : (index, index) -> index
    %373 = "emitc.subscript"(%7, %372) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %374 = "emitc.apply"(%373) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%374, %304, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
}) : () -> ()
