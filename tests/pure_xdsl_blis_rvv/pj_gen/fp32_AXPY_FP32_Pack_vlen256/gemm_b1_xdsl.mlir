"builtin.module"() ({
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_16x15_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
    %27 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %28 = "emitc.apply"(%27) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %29 = "emitc.call_opaque"(%28, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %30 = "emitc.add"(%10, %18) : (index, index) -> index
    %31 = "emitc.subscript"(%7, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %32 = "emitc.apply"(%31) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %33 = "emitc.call_opaque"(%32, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %34 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %35 = "emitc.apply"(%34) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%35, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %37 = "emitc.add"(%8, %18) : (index, index) -> index
    %38 = "emitc.subscript"(%7, %37) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %39 = "emitc.apply"(%38) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%39, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %41 = "emitc.mul"(%12, %8) : (index, index) -> index
    %42 = "emitc.subscript"(%7, %41) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %43 = "emitc.apply"(%42) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %44 = "emitc.call_opaque"(%43, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %45 = "emitc.add"(%41, %18) : (index, index) -> index
    %46 = "emitc.subscript"(%7, %45) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %47 = "emitc.apply"(%46) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %48 = "emitc.call_opaque"(%47, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %49 = "emitc.mul"(%13, %8) : (index, index) -> index
    %50 = "emitc.subscript"(%7, %49) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %51 = "emitc.apply"(%50) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %52 = "emitc.call_opaque"(%51, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %53 = "emitc.add"(%49, %18) : (index, index) -> index
    %54 = "emitc.subscript"(%7, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %55 = "emitc.apply"(%54) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %56 = "emitc.call_opaque"(%55, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %57 = "emitc.mul"(%14, %8) : (index, index) -> index
    %58 = "emitc.subscript"(%7, %57) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %59 = "emitc.apply"(%58) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %60 = "emitc.call_opaque"(%59, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %61 = "emitc.add"(%57, %18) : (index, index) -> index
    %62 = "emitc.subscript"(%7, %61) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %63 = "emitc.apply"(%62) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %64 = "emitc.call_opaque"(%63, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %65 = "emitc.mul"(%15, %8) : (index, index) -> index
    %66 = "emitc.subscript"(%7, %65) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %67 = "emitc.apply"(%66) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %68 = "emitc.call_opaque"(%67, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %69 = "emitc.add"(%65, %18) : (index, index) -> index
    %70 = "emitc.subscript"(%7, %69) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %71 = "emitc.apply"(%70) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %72 = "emitc.call_opaque"(%71, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %73 = "emitc.mul"(%16, %8) : (index, index) -> index
    %74 = "emitc.subscript"(%7, %73) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %75 = "emitc.apply"(%74) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %76 = "emitc.call_opaque"(%75, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %77 = "emitc.add"(%73, %18) : (index, index) -> index
    %78 = "emitc.subscript"(%7, %77) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %79 = "emitc.apply"(%78) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %80 = "emitc.call_opaque"(%79, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %81 = "emitc.mul"(%17, %8) : (index, index) -> index
    %82 = "emitc.subscript"(%7, %81) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %83 = "emitc.apply"(%82) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %84 = "emitc.call_opaque"(%83, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %85 = "emitc.add"(%81, %18) : (index, index) -> index
    %86 = "emitc.subscript"(%7, %85) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %87 = "emitc.apply"(%86) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %88 = "emitc.call_opaque"(%87, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %89 = "emitc.mul"(%18, %8) : (index, index) -> index
    %90 = "emitc.subscript"(%7, %89) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %91 = "emitc.apply"(%90) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %92 = "emitc.call_opaque"(%91, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %93 = "emitc.add"(%89, %18) : (index, index) -> index
    %94 = "emitc.subscript"(%7, %93) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %95 = "emitc.apply"(%94) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %96 = "emitc.call_opaque"(%95, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %97 = "emitc.mul"(%19, %8) : (index, index) -> index
    %98 = "emitc.subscript"(%7, %97) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %99 = "emitc.apply"(%98) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %100 = "emitc.call_opaque"(%99, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %101 = "emitc.add"(%97, %18) : (index, index) -> index
    %102 = "emitc.subscript"(%7, %101) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %103 = "emitc.apply"(%102) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %104 = "emitc.call_opaque"(%103, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %105 = "emitc.mul"(%20, %8) : (index, index) -> index
    %106 = "emitc.subscript"(%7, %105) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %107 = "emitc.apply"(%106) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %108 = "emitc.call_opaque"(%107, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %109 = "emitc.add"(%105, %18) : (index, index) -> index
    %110 = "emitc.subscript"(%7, %109) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %111 = "emitc.apply"(%110) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %112 = "emitc.call_opaque"(%111, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %113 = "emitc.mul"(%21, %8) : (index, index) -> index
    %114 = "emitc.subscript"(%7, %113) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %115 = "emitc.apply"(%114) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %116 = "emitc.call_opaque"(%115, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %117 = "emitc.add"(%113, %18) : (index, index) -> index
    %118 = "emitc.subscript"(%7, %117) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %119 = "emitc.apply"(%118) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %120 = "emitc.call_opaque"(%119, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %121 = "emitc.mul"(%22, %8) : (index, index) -> index
    %122 = "emitc.subscript"(%7, %121) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %123 = "emitc.apply"(%122) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %124 = "emitc.call_opaque"(%123, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %125 = "emitc.add"(%121, %18) : (index, index) -> index
    %126 = "emitc.subscript"(%7, %125) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %127 = "emitc.apply"(%126) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %128 = "emitc.call_opaque"(%127, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %129 = "emitc.mul"(%23, %8) : (index, index) -> index
    %130 = "emitc.subscript"(%7, %129) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %131 = "emitc.apply"(%130) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %132 = "emitc.call_opaque"(%131, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %133 = "emitc.add"(%129, %18) : (index, index) -> index
    %134 = "emitc.subscript"(%7, %133) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %135 = "emitc.apply"(%134) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %136 = "emitc.call_opaque"(%135, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %137 = "emitc.mul"(%24, %8) : (index, index) -> index
    %138 = "emitc.subscript"(%7, %137) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %139 = "emitc.apply"(%138) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %140 = "emitc.call_opaque"(%139, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %141 = "emitc.add"(%137, %18) : (index, index) -> index
    %142 = "emitc.subscript"(%7, %141) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %143 = "emitc.apply"(%142) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %144 = "emitc.call_opaque"(%143, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %145 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%145, %29) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %146 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%146, %33) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %147 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%147, %36) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %148 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%148, %40) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %149 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%149, %44) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %150 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%150, %48) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %151 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%151, %52) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %152 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%152, %56) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %153 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%153, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %154 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%154, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %155 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%155, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %156 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%156, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %157 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%157, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %158 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%158, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %159 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%159, %84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %160 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%160, %88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %161 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%161, %92) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %162 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%162, %96) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %163 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%163, %100) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %164 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%164, %104) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %165 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%165, %108) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %166 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%166, %112) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %167 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%167, %116) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %168 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%168, %120) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %169 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%169, %124) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %170 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%170, %128) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %171 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%171, %132) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %172 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%172, %136) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %173 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%173, %140) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %174 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%174, %144) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%175 : index):
      %176 = "emitc.mul"(%175, %3) : (index, index) -> index
      %177 = "emitc.constant"() {value = 8 : index} : () -> index
      %178 = "emitc.subscript"(%2, %176) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %179 = "emitc.apply"(%178) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %180 = "emitc.call_opaque"(%179, %177) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %181 = "emitc.add"(%176, %18) : (index, index) -> index
      %182 = "emitc.subscript"(%2, %181) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %183 = "emitc.apply"(%182) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %184 = "emitc.call_opaque"(%183, %177) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %185 = "emitc.mul"(%175, %5) : (index, index) -> index
      %186 = "emitc.add"(%185, %11) : (index, index) -> index
      %187 = "emitc.add"(%185, %12) : (index, index) -> index
      %188 = "emitc.add"(%185, %13) : (index, index) -> index
      %189 = "emitc.add"(%185, %14) : (index, index) -> index
      %190 = "emitc.add"(%185, %15) : (index, index) -> index
      %191 = "emitc.add"(%185, %16) : (index, index) -> index
      %192 = "emitc.add"(%185, %17) : (index, index) -> index
      %193 = "emitc.add"(%185, %18) : (index, index) -> index
      %194 = "emitc.add"(%185, %19) : (index, index) -> index
      %195 = "emitc.add"(%185, %20) : (index, index) -> index
      %196 = "emitc.add"(%185, %21) : (index, index) -> index
      %197 = "emitc.add"(%185, %22) : (index, index) -> index
      %198 = "emitc.add"(%185, %23) : (index, index) -> index
      %199 = "emitc.add"(%185, %24) : (index, index) -> index
      %200 = "emitc.load"(%145) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %201 = "emitc.subscript"(%4, %185) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %202 = "emitc.load"(%201) : (!emitc.lvalue<f32>) -> f32
      %203 = "emitc.call_opaque"(%200, %202, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %204 = "emitc.load"(%146) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %205 = "emitc.subscript"(%4, %185) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %206 = "emitc.load"(%205) : (!emitc.lvalue<f32>) -> f32
      %207 = "emitc.call_opaque"(%204, %206, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %208 = "emitc.load"(%147) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %209 = "emitc.subscript"(%4, %186) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %210 = "emitc.load"(%209) : (!emitc.lvalue<f32>) -> f32
      %211 = "emitc.call_opaque"(%208, %210, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %212 = "emitc.load"(%148) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %213 = "emitc.subscript"(%4, %186) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %214 = "emitc.load"(%213) : (!emitc.lvalue<f32>) -> f32
      %215 = "emitc.call_opaque"(%212, %214, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %216 = "emitc.load"(%149) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %217 = "emitc.subscript"(%4, %187) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %218 = "emitc.load"(%217) : (!emitc.lvalue<f32>) -> f32
      %219 = "emitc.call_opaque"(%216, %218, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %220 = "emitc.load"(%150) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %221 = "emitc.subscript"(%4, %187) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %222 = "emitc.load"(%221) : (!emitc.lvalue<f32>) -> f32
      %223 = "emitc.call_opaque"(%220, %222, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %224 = "emitc.load"(%151) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %225 = "emitc.subscript"(%4, %188) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %226 = "emitc.load"(%225) : (!emitc.lvalue<f32>) -> f32
      %227 = "emitc.call_opaque"(%224, %226, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %228 = "emitc.load"(%152) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %229 = "emitc.subscript"(%4, %188) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %230 = "emitc.load"(%229) : (!emitc.lvalue<f32>) -> f32
      %231 = "emitc.call_opaque"(%228, %230, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %232 = "emitc.load"(%153) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %233 = "emitc.subscript"(%4, %189) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %234 = "emitc.load"(%233) : (!emitc.lvalue<f32>) -> f32
      %235 = "emitc.call_opaque"(%232, %234, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %236 = "emitc.load"(%154) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %237 = "emitc.subscript"(%4, %189) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %238 = "emitc.load"(%237) : (!emitc.lvalue<f32>) -> f32
      %239 = "emitc.call_opaque"(%236, %238, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %240 = "emitc.load"(%155) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %241 = "emitc.subscript"(%4, %190) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %242 = "emitc.load"(%241) : (!emitc.lvalue<f32>) -> f32
      %243 = "emitc.call_opaque"(%240, %242, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %244 = "emitc.load"(%156) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %245 = "emitc.subscript"(%4, %190) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %246 = "emitc.load"(%245) : (!emitc.lvalue<f32>) -> f32
      %247 = "emitc.call_opaque"(%244, %246, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %248 = "emitc.load"(%157) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %249 = "emitc.subscript"(%4, %191) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %250 = "emitc.load"(%249) : (!emitc.lvalue<f32>) -> f32
      %251 = "emitc.call_opaque"(%248, %250, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %252 = "emitc.load"(%158) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %253 = "emitc.subscript"(%4, %191) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %254 = "emitc.load"(%253) : (!emitc.lvalue<f32>) -> f32
      %255 = "emitc.call_opaque"(%252, %254, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %256 = "emitc.load"(%159) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %257 = "emitc.subscript"(%4, %192) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %258 = "emitc.load"(%257) : (!emitc.lvalue<f32>) -> f32
      %259 = "emitc.call_opaque"(%256, %258, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %260 = "emitc.load"(%160) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %261 = "emitc.subscript"(%4, %192) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %262 = "emitc.load"(%261) : (!emitc.lvalue<f32>) -> f32
      %263 = "emitc.call_opaque"(%260, %262, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %264 = "emitc.load"(%161) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %265 = "emitc.subscript"(%4, %193) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %266 = "emitc.load"(%265) : (!emitc.lvalue<f32>) -> f32
      %267 = "emitc.call_opaque"(%264, %266, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %268 = "emitc.load"(%162) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %269 = "emitc.subscript"(%4, %193) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %270 = "emitc.load"(%269) : (!emitc.lvalue<f32>) -> f32
      %271 = "emitc.call_opaque"(%268, %270, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %272 = "emitc.load"(%163) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %273 = "emitc.subscript"(%4, %194) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %274 = "emitc.load"(%273) : (!emitc.lvalue<f32>) -> f32
      %275 = "emitc.call_opaque"(%272, %274, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %276 = "emitc.load"(%164) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %277 = "emitc.subscript"(%4, %194) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %278 = "emitc.load"(%277) : (!emitc.lvalue<f32>) -> f32
      %279 = "emitc.call_opaque"(%276, %278, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %280 = "emitc.load"(%165) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %281 = "emitc.subscript"(%4, %195) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %282 = "emitc.load"(%281) : (!emitc.lvalue<f32>) -> f32
      %283 = "emitc.call_opaque"(%280, %282, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %284 = "emitc.load"(%166) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %285 = "emitc.subscript"(%4, %195) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %286 = "emitc.load"(%285) : (!emitc.lvalue<f32>) -> f32
      %287 = "emitc.call_opaque"(%284, %286, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %288 = "emitc.load"(%167) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %289 = "emitc.subscript"(%4, %196) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %290 = "emitc.load"(%289) : (!emitc.lvalue<f32>) -> f32
      %291 = "emitc.call_opaque"(%288, %290, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %292 = "emitc.load"(%168) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %293 = "emitc.subscript"(%4, %196) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %294 = "emitc.load"(%293) : (!emitc.lvalue<f32>) -> f32
      %295 = "emitc.call_opaque"(%292, %294, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %296 = "emitc.load"(%169) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %297 = "emitc.subscript"(%4, %197) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %298 = "emitc.load"(%297) : (!emitc.lvalue<f32>) -> f32
      %299 = "emitc.call_opaque"(%296, %298, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %300 = "emitc.load"(%170) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %301 = "emitc.subscript"(%4, %197) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %302 = "emitc.load"(%301) : (!emitc.lvalue<f32>) -> f32
      %303 = "emitc.call_opaque"(%300, %302, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %304 = "emitc.load"(%171) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %305 = "emitc.subscript"(%4, %198) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %306 = "emitc.load"(%305) : (!emitc.lvalue<f32>) -> f32
      %307 = "emitc.call_opaque"(%304, %306, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %308 = "emitc.load"(%172) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %309 = "emitc.subscript"(%4, %198) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %310 = "emitc.load"(%309) : (!emitc.lvalue<f32>) -> f32
      %311 = "emitc.call_opaque"(%308, %310, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %312 = "emitc.load"(%173) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %313 = "emitc.subscript"(%4, %199) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %314 = "emitc.load"(%313) : (!emitc.lvalue<f32>) -> f32
      %315 = "emitc.call_opaque"(%312, %314, %180, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %316 = "emitc.load"(%174) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %317 = "emitc.subscript"(%4, %199) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %318 = "emitc.load"(%317) : (!emitc.lvalue<f32>) -> f32
      %319 = "emitc.call_opaque"(%316, %318, %184, %177) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%145, %203) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%146, %207) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%147, %211) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%148, %215) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%149, %219) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%150, %223) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%151, %227) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%152, %231) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%153, %235) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%154, %239) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%155, %243) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%156, %247) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%157, %251) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%158, %255) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%159, %259) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%160, %263) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%161, %267) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%162, %271) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%163, %275) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%164, %279) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%165, %283) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%166, %287) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%167, %291) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%168, %295) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%169, %299) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%170, %303) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%171, %307) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%172, %311) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%173, %315) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%174, %319) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %320 = "emitc.load"(%145) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %321 = "emitc.load"(%146) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %322 = "emitc.load"(%147) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %323 = "emitc.load"(%148) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %324 = "emitc.load"(%149) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %325 = "emitc.load"(%150) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %326 = "emitc.load"(%151) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %327 = "emitc.load"(%152) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %328 = "emitc.load"(%153) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %329 = "emitc.load"(%154) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %330 = "emitc.load"(%155) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %331 = "emitc.load"(%156) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %332 = "emitc.load"(%157) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %333 = "emitc.load"(%158) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %334 = "emitc.load"(%159) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %335 = "emitc.load"(%160) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %336 = "emitc.load"(%161) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %337 = "emitc.load"(%162) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %338 = "emitc.load"(%163) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %339 = "emitc.load"(%164) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %340 = "emitc.load"(%165) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %341 = "emitc.load"(%166) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %342 = "emitc.load"(%167) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %343 = "emitc.load"(%168) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %344 = "emitc.load"(%169) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %345 = "emitc.load"(%170) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %346 = "emitc.load"(%171) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %347 = "emitc.load"(%172) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %348 = "emitc.load"(%173) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %349 = "emitc.load"(%174) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %350 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %351 = "emitc.apply"(%350) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%351, %320, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %352 = "emitc.add"(%10, %18) : (index, index) -> index
    %353 = "emitc.subscript"(%7, %352) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %354 = "emitc.apply"(%353) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%354, %321, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %355 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %356 = "emitc.apply"(%355) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%356, %322, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %357 = "emitc.add"(%8, %18) : (index, index) -> index
    %358 = "emitc.subscript"(%7, %357) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %359 = "emitc.apply"(%358) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%359, %323, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %360 = "emitc.mul"(%12, %8) : (index, index) -> index
    %361 = "emitc.subscript"(%7, %360) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %362 = "emitc.apply"(%361) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%362, %324, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %363 = "emitc.add"(%360, %18) : (index, index) -> index
    %364 = "emitc.subscript"(%7, %363) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %365 = "emitc.apply"(%364) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%365, %325, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %366 = "emitc.mul"(%13, %8) : (index, index) -> index
    %367 = "emitc.subscript"(%7, %366) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %368 = "emitc.apply"(%367) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%368, %326, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %369 = "emitc.add"(%366, %18) : (index, index) -> index
    %370 = "emitc.subscript"(%7, %369) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %371 = "emitc.apply"(%370) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%371, %327, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %372 = "emitc.mul"(%14, %8) : (index, index) -> index
    %373 = "emitc.subscript"(%7, %372) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %374 = "emitc.apply"(%373) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%374, %328, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %375 = "emitc.add"(%372, %18) : (index, index) -> index
    %376 = "emitc.subscript"(%7, %375) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %377 = "emitc.apply"(%376) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%377, %329, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %378 = "emitc.mul"(%15, %8) : (index, index) -> index
    %379 = "emitc.subscript"(%7, %378) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %380 = "emitc.apply"(%379) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%380, %330, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %381 = "emitc.add"(%378, %18) : (index, index) -> index
    %382 = "emitc.subscript"(%7, %381) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %383 = "emitc.apply"(%382) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%383, %331, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %384 = "emitc.mul"(%16, %8) : (index, index) -> index
    %385 = "emitc.subscript"(%7, %384) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %386 = "emitc.apply"(%385) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%386, %332, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %387 = "emitc.add"(%384, %18) : (index, index) -> index
    %388 = "emitc.subscript"(%7, %387) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %389 = "emitc.apply"(%388) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%389, %333, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %390 = "emitc.mul"(%17, %8) : (index, index) -> index
    %391 = "emitc.subscript"(%7, %390) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %392 = "emitc.apply"(%391) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%392, %334, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %393 = "emitc.add"(%390, %18) : (index, index) -> index
    %394 = "emitc.subscript"(%7, %393) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %395 = "emitc.apply"(%394) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%395, %335, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %396 = "emitc.mul"(%18, %8) : (index, index) -> index
    %397 = "emitc.subscript"(%7, %396) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %398 = "emitc.apply"(%397) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%398, %336, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %399 = "emitc.add"(%396, %18) : (index, index) -> index
    %400 = "emitc.subscript"(%7, %399) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %401 = "emitc.apply"(%400) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%401, %337, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %402 = "emitc.mul"(%19, %8) : (index, index) -> index
    %403 = "emitc.subscript"(%7, %402) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %404 = "emitc.apply"(%403) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%404, %338, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %405 = "emitc.add"(%402, %18) : (index, index) -> index
    %406 = "emitc.subscript"(%7, %405) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %407 = "emitc.apply"(%406) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%407, %339, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %408 = "emitc.mul"(%20, %8) : (index, index) -> index
    %409 = "emitc.subscript"(%7, %408) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %410 = "emitc.apply"(%409) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%410, %340, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %411 = "emitc.add"(%408, %18) : (index, index) -> index
    %412 = "emitc.subscript"(%7, %411) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %413 = "emitc.apply"(%412) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%413, %341, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %414 = "emitc.mul"(%21, %8) : (index, index) -> index
    %415 = "emitc.subscript"(%7, %414) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %416 = "emitc.apply"(%415) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%416, %342, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %417 = "emitc.add"(%414, %18) : (index, index) -> index
    %418 = "emitc.subscript"(%7, %417) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %419 = "emitc.apply"(%418) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%419, %343, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %420 = "emitc.mul"(%22, %8) : (index, index) -> index
    %421 = "emitc.subscript"(%7, %420) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %422 = "emitc.apply"(%421) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%422, %344, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %423 = "emitc.add"(%420, %18) : (index, index) -> index
    %424 = "emitc.subscript"(%7, %423) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %425 = "emitc.apply"(%424) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%425, %345, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %426 = "emitc.mul"(%23, %8) : (index, index) -> index
    %427 = "emitc.subscript"(%7, %426) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %428 = "emitc.apply"(%427) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%428, %346, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %429 = "emitc.add"(%426, %18) : (index, index) -> index
    %430 = "emitc.subscript"(%7, %429) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %431 = "emitc.apply"(%430) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%431, %347, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %432 = "emitc.mul"(%24, %8) : (index, index) -> index
    %433 = "emitc.subscript"(%7, %432) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %434 = "emitc.apply"(%433) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%434, %348, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %435 = "emitc.add"(%432, %18) : (index, index) -> index
    %436 = "emitc.subscript"(%7, %435) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %437 = "emitc.apply"(%436) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%437, %349, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
}) : () -> ()