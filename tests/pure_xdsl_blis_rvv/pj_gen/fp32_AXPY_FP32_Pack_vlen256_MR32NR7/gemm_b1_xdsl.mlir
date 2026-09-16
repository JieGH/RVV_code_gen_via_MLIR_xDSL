"builtin.module"() ({
  "func.func"() <{sym_name = "gemm_microkernel_v_fp32_32x7_b1", function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()}> ({
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
    %137 = "emitc.mul"(%16, %8) : (index, index) -> index
    %138 = "emitc.subscript"(%7, %137) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %139 = "emitc.apply"(%138) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %140 = "emitc.call_opaque"(%139, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %141 = "emitc.add"(%137, %18) : (index, index) -> index
    %142 = "emitc.subscript"(%7, %141) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %143 = "emitc.apply"(%142) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %144 = "emitc.call_opaque"(%143, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %145 = "emitc.add"(%137, %26) : (index, index) -> index
    %146 = "emitc.subscript"(%7, %145) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %147 = "emitc.apply"(%146) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %148 = "emitc.call_opaque"(%147, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %149 = "emitc.add"(%137, %34) : (index, index) -> index
    %150 = "emitc.subscript"(%7, %149) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %151 = "emitc.apply"(%150) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %152 = "emitc.call_opaque"(%151, %18) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
    %153 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%153, %45) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %154 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%154, %49) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %155 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%155, %53) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %156 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%156, %57) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %157 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%157, %60) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %158 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%158, %64) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %159 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%159, %68) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %160 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%160, %72) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %161 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%161, %76) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %162 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%162, %80) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %163 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%163, %84) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %164 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%164, %88) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %165 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%165, %92) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %166 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%166, %96) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %167 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%167, %100) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %168 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%168, %104) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %169 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%169, %108) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %170 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%170, %112) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %171 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%171, %116) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %172 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%172, %120) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %173 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%173, %124) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %174 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%174, %128) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %175 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%175, %132) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %176 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%176, %136) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %177 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%177, %140) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %178 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%178, %144) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %179 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%179, %148) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    %180 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
    "emitc.assign"(%180, %152) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
    "emitc.for"(%10, %0, %11) ({
    ^bb1(%181 : index):
      %182 = "emitc.mul"(%181, %3) : (index, index) -> index
      %183 = "emitc.constant"() {value = 8 : index} : () -> index
      %184 = "emitc.subscript"(%2, %182) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %185 = "emitc.apply"(%184) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %186 = "emitc.call_opaque"(%185, %183) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %187 = "emitc.add"(%182, %18) : (index, index) -> index
      %188 = "emitc.subscript"(%2, %187) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %189 = "emitc.apply"(%188) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %190 = "emitc.call_opaque"(%189, %183) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %191 = "emitc.add"(%182, %26) : (index, index) -> index
      %192 = "emitc.subscript"(%2, %191) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %193 = "emitc.apply"(%192) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %194 = "emitc.call_opaque"(%193, %183) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %195 = "emitc.add"(%182, %34) : (index, index) -> index
      %196 = "emitc.subscript"(%2, %195) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %197 = "emitc.apply"(%196) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
      %198 = "emitc.call_opaque"(%197, %183) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
      %199 = "emitc.mul"(%181, %5) : (index, index) -> index
      %200 = "emitc.add"(%199, %11) : (index, index) -> index
      %201 = "emitc.add"(%199, %12) : (index, index) -> index
      %202 = "emitc.add"(%199, %13) : (index, index) -> index
      %203 = "emitc.add"(%199, %14) : (index, index) -> index
      %204 = "emitc.add"(%199, %15) : (index, index) -> index
      %205 = "emitc.add"(%199, %16) : (index, index) -> index
      %206 = "emitc.load"(%153) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %207 = "emitc.subscript"(%4, %199) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %208 = "emitc.load"(%207) : (!emitc.lvalue<f32>) -> f32
      %209 = "emitc.call_opaque"(%206, %208, %186, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %210 = "emitc.load"(%154) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %211 = "emitc.subscript"(%4, %199) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %212 = "emitc.load"(%211) : (!emitc.lvalue<f32>) -> f32
      %213 = "emitc.call_opaque"(%210, %212, %190, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %214 = "emitc.load"(%155) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %215 = "emitc.subscript"(%4, %199) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %216 = "emitc.load"(%215) : (!emitc.lvalue<f32>) -> f32
      %217 = "emitc.call_opaque"(%214, %216, %194, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %218 = "emitc.load"(%156) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %219 = "emitc.subscript"(%4, %199) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %220 = "emitc.load"(%219) : (!emitc.lvalue<f32>) -> f32
      %221 = "emitc.call_opaque"(%218, %220, %198, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %222 = "emitc.load"(%157) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %223 = "emitc.subscript"(%4, %200) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %224 = "emitc.load"(%223) : (!emitc.lvalue<f32>) -> f32
      %225 = "emitc.call_opaque"(%222, %224, %186, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %226 = "emitc.load"(%158) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %227 = "emitc.subscript"(%4, %200) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %228 = "emitc.load"(%227) : (!emitc.lvalue<f32>) -> f32
      %229 = "emitc.call_opaque"(%226, %228, %190, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %230 = "emitc.load"(%159) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %231 = "emitc.subscript"(%4, %200) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %232 = "emitc.load"(%231) : (!emitc.lvalue<f32>) -> f32
      %233 = "emitc.call_opaque"(%230, %232, %194, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %234 = "emitc.load"(%160) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %235 = "emitc.subscript"(%4, %200) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %236 = "emitc.load"(%235) : (!emitc.lvalue<f32>) -> f32
      %237 = "emitc.call_opaque"(%234, %236, %198, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %238 = "emitc.load"(%161) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %239 = "emitc.subscript"(%4, %201) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %240 = "emitc.load"(%239) : (!emitc.lvalue<f32>) -> f32
      %241 = "emitc.call_opaque"(%238, %240, %186, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %242 = "emitc.load"(%162) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %243 = "emitc.subscript"(%4, %201) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %244 = "emitc.load"(%243) : (!emitc.lvalue<f32>) -> f32
      %245 = "emitc.call_opaque"(%242, %244, %190, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %246 = "emitc.load"(%163) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %247 = "emitc.subscript"(%4, %201) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %248 = "emitc.load"(%247) : (!emitc.lvalue<f32>) -> f32
      %249 = "emitc.call_opaque"(%246, %248, %194, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %250 = "emitc.load"(%164) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %251 = "emitc.subscript"(%4, %201) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %252 = "emitc.load"(%251) : (!emitc.lvalue<f32>) -> f32
      %253 = "emitc.call_opaque"(%250, %252, %198, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %254 = "emitc.load"(%165) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %255 = "emitc.subscript"(%4, %202) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %256 = "emitc.load"(%255) : (!emitc.lvalue<f32>) -> f32
      %257 = "emitc.call_opaque"(%254, %256, %186, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %258 = "emitc.load"(%166) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %259 = "emitc.subscript"(%4, %202) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %260 = "emitc.load"(%259) : (!emitc.lvalue<f32>) -> f32
      %261 = "emitc.call_opaque"(%258, %260, %190, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %262 = "emitc.load"(%167) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %263 = "emitc.subscript"(%4, %202) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %264 = "emitc.load"(%263) : (!emitc.lvalue<f32>) -> f32
      %265 = "emitc.call_opaque"(%262, %264, %194, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %266 = "emitc.load"(%168) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %267 = "emitc.subscript"(%4, %202) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %268 = "emitc.load"(%267) : (!emitc.lvalue<f32>) -> f32
      %269 = "emitc.call_opaque"(%266, %268, %198, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %270 = "emitc.load"(%169) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %271 = "emitc.subscript"(%4, %203) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %272 = "emitc.load"(%271) : (!emitc.lvalue<f32>) -> f32
      %273 = "emitc.call_opaque"(%270, %272, %186, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %274 = "emitc.load"(%170) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %275 = "emitc.subscript"(%4, %203) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %276 = "emitc.load"(%275) : (!emitc.lvalue<f32>) -> f32
      %277 = "emitc.call_opaque"(%274, %276, %190, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %278 = "emitc.load"(%171) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %279 = "emitc.subscript"(%4, %203) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %280 = "emitc.load"(%279) : (!emitc.lvalue<f32>) -> f32
      %281 = "emitc.call_opaque"(%278, %280, %194, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %282 = "emitc.load"(%172) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %283 = "emitc.subscript"(%4, %203) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %284 = "emitc.load"(%283) : (!emitc.lvalue<f32>) -> f32
      %285 = "emitc.call_opaque"(%282, %284, %198, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %286 = "emitc.load"(%173) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %287 = "emitc.subscript"(%4, %204) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %288 = "emitc.load"(%287) : (!emitc.lvalue<f32>) -> f32
      %289 = "emitc.call_opaque"(%286, %288, %186, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %290 = "emitc.load"(%174) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %291 = "emitc.subscript"(%4, %204) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %292 = "emitc.load"(%291) : (!emitc.lvalue<f32>) -> f32
      %293 = "emitc.call_opaque"(%290, %292, %190, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %294 = "emitc.load"(%175) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %295 = "emitc.subscript"(%4, %204) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %296 = "emitc.load"(%295) : (!emitc.lvalue<f32>) -> f32
      %297 = "emitc.call_opaque"(%294, %296, %194, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %298 = "emitc.load"(%176) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %299 = "emitc.subscript"(%4, %204) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %300 = "emitc.load"(%299) : (!emitc.lvalue<f32>) -> f32
      %301 = "emitc.call_opaque"(%298, %300, %198, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %302 = "emitc.load"(%177) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %303 = "emitc.subscript"(%4, %205) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %304 = "emitc.load"(%303) : (!emitc.lvalue<f32>) -> f32
      %305 = "emitc.call_opaque"(%302, %304, %186, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %306 = "emitc.load"(%178) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %307 = "emitc.subscript"(%4, %205) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %308 = "emitc.load"(%307) : (!emitc.lvalue<f32>) -> f32
      %309 = "emitc.call_opaque"(%306, %308, %190, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %310 = "emitc.load"(%179) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %311 = "emitc.subscript"(%4, %205) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %312 = "emitc.load"(%311) : (!emitc.lvalue<f32>) -> f32
      %313 = "emitc.call_opaque"(%310, %312, %194, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      %314 = "emitc.load"(%180) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
      %315 = "emitc.subscript"(%4, %205) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
      %316 = "emitc.load"(%315) : (!emitc.lvalue<f32>) -> f32
      %317 = "emitc.call_opaque"(%314, %316, %198, %183) <{callee = "__riscv_vfmacc_vf_f32m1"}> : (!emitc.opaque<"vfloat32m1_t">, f32, !emitc.opaque<"vfloat32m1_t">, index) -> !emitc.opaque<"vfloat32m1_t">
      "emitc.assign"(%153, %209) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%154, %213) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%155, %217) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%156, %221) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%157, %225) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%158, %229) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%159, %233) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%160, %237) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%161, %241) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%162, %245) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%163, %249) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%164, %253) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%165, %257) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%166, %261) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%167, %265) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%168, %269) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%169, %273) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%170, %277) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%171, %281) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%172, %285) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%173, %289) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%174, %293) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%175, %297) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%176, %301) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%177, %305) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%178, %309) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%179, %313) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.assign"(%180, %317) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>, !emitc.opaque<"vfloat32m1_t">) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %318 = "emitc.load"(%153) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %319 = "emitc.load"(%154) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %320 = "emitc.load"(%155) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %321 = "emitc.load"(%156) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %322 = "emitc.load"(%157) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %323 = "emitc.load"(%158) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %324 = "emitc.load"(%159) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %325 = "emitc.load"(%160) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %326 = "emitc.load"(%161) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %327 = "emitc.load"(%162) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %328 = "emitc.load"(%163) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %329 = "emitc.load"(%164) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %330 = "emitc.load"(%165) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %331 = "emitc.load"(%166) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %332 = "emitc.load"(%167) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %333 = "emitc.load"(%168) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %334 = "emitc.load"(%169) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %335 = "emitc.load"(%170) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %336 = "emitc.load"(%171) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %337 = "emitc.load"(%172) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %338 = "emitc.load"(%173) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %339 = "emitc.load"(%174) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %340 = "emitc.load"(%175) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %341 = "emitc.load"(%176) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %342 = "emitc.load"(%177) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %343 = "emitc.load"(%178) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %344 = "emitc.load"(%179) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %345 = "emitc.load"(%180) : (!emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>) -> !emitc.opaque<"vfloat32m1_t">
    %346 = "emitc.subscript"(%7, %10) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %347 = "emitc.apply"(%346) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%347, %318, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %348 = "emitc.add"(%10, %18) : (index, index) -> index
    %349 = "emitc.subscript"(%7, %348) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %350 = "emitc.apply"(%349) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%350, %319, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %351 = "emitc.add"(%10, %26) : (index, index) -> index
    %352 = "emitc.subscript"(%7, %351) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %353 = "emitc.apply"(%352) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%353, %320, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %354 = "emitc.add"(%10, %34) : (index, index) -> index
    %355 = "emitc.subscript"(%7, %354) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %356 = "emitc.apply"(%355) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%356, %321, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %357 = "emitc.subscript"(%7, %8) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %358 = "emitc.apply"(%357) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%358, %322, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %359 = "emitc.add"(%8, %18) : (index, index) -> index
    %360 = "emitc.subscript"(%7, %359) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %361 = "emitc.apply"(%360) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%361, %323, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %362 = "emitc.add"(%8, %26) : (index, index) -> index
    %363 = "emitc.subscript"(%7, %362) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %364 = "emitc.apply"(%363) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%364, %324, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %365 = "emitc.add"(%8, %34) : (index, index) -> index
    %366 = "emitc.subscript"(%7, %365) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %367 = "emitc.apply"(%366) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%367, %325, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %368 = "emitc.mul"(%12, %8) : (index, index) -> index
    %369 = "emitc.subscript"(%7, %368) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %370 = "emitc.apply"(%369) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%370, %326, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %371 = "emitc.add"(%368, %18) : (index, index) -> index
    %372 = "emitc.subscript"(%7, %371) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %373 = "emitc.apply"(%372) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%373, %327, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %374 = "emitc.add"(%368, %26) : (index, index) -> index
    %375 = "emitc.subscript"(%7, %374) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %376 = "emitc.apply"(%375) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%376, %328, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %377 = "emitc.add"(%368, %34) : (index, index) -> index
    %378 = "emitc.subscript"(%7, %377) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %379 = "emitc.apply"(%378) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%379, %329, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %380 = "emitc.mul"(%13, %8) : (index, index) -> index
    %381 = "emitc.subscript"(%7, %380) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %382 = "emitc.apply"(%381) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%382, %330, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %383 = "emitc.add"(%380, %18) : (index, index) -> index
    %384 = "emitc.subscript"(%7, %383) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %385 = "emitc.apply"(%384) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%385, %331, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %386 = "emitc.add"(%380, %26) : (index, index) -> index
    %387 = "emitc.subscript"(%7, %386) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %388 = "emitc.apply"(%387) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%388, %332, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %389 = "emitc.add"(%380, %34) : (index, index) -> index
    %390 = "emitc.subscript"(%7, %389) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %391 = "emitc.apply"(%390) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%391, %333, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %392 = "emitc.mul"(%14, %8) : (index, index) -> index
    %393 = "emitc.subscript"(%7, %392) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %394 = "emitc.apply"(%393) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%394, %334, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %395 = "emitc.add"(%392, %18) : (index, index) -> index
    %396 = "emitc.subscript"(%7, %395) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %397 = "emitc.apply"(%396) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%397, %335, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %398 = "emitc.add"(%392, %26) : (index, index) -> index
    %399 = "emitc.subscript"(%7, %398) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %400 = "emitc.apply"(%399) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%400, %336, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %401 = "emitc.add"(%392, %34) : (index, index) -> index
    %402 = "emitc.subscript"(%7, %401) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %403 = "emitc.apply"(%402) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%403, %337, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %404 = "emitc.mul"(%15, %8) : (index, index) -> index
    %405 = "emitc.subscript"(%7, %404) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %406 = "emitc.apply"(%405) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%406, %338, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %407 = "emitc.add"(%404, %18) : (index, index) -> index
    %408 = "emitc.subscript"(%7, %407) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %409 = "emitc.apply"(%408) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%409, %339, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %410 = "emitc.add"(%404, %26) : (index, index) -> index
    %411 = "emitc.subscript"(%7, %410) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %412 = "emitc.apply"(%411) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%412, %340, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %413 = "emitc.add"(%404, %34) : (index, index) -> index
    %414 = "emitc.subscript"(%7, %413) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %415 = "emitc.apply"(%414) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%415, %341, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %416 = "emitc.mul"(%16, %8) : (index, index) -> index
    %417 = "emitc.subscript"(%7, %416) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %418 = "emitc.apply"(%417) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%418, %342, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %419 = "emitc.add"(%416, %18) : (index, index) -> index
    %420 = "emitc.subscript"(%7, %419) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %421 = "emitc.apply"(%420) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%421, %343, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %422 = "emitc.add"(%416, %26) : (index, index) -> index
    %423 = "emitc.subscript"(%7, %422) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %424 = "emitc.apply"(%423) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%424, %344, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    %425 = "emitc.add"(%416, %34) : (index, index) -> index
    %426 = "emitc.subscript"(%7, %425) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    %427 = "emitc.apply"(%426) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    "emitc.call_opaque"(%427, %345, %18) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, !emitc.ptr<f32>, !emitc.ptr<f32>, index, !emitc.ptr<f32>, index, !emitc.ptr<f32>, !emitc.ptr<f32>, index) -> f32} : () -> ()
}) : () -> ()
