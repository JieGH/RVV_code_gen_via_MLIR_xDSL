"builtin.module"() ({
  "func.func"() <{sym_name = "test_mixed_precision", function_type = (!emitc.ptr<i8>, !emitc.ptr<i8>, !emitc.ptr<i32>, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<i8>, %1 : !emitc.ptr<i8>, %2 : !emitc.ptr<i32>, %3 : index):
    %4 = "emitc.constant"() {value = 0 : index} : () -> index
    %5 = "emitc.constant"() {value = 0 : i16} : () -> i16
    %6 = "emitc.constant"() {value = 0 : i32} : () -> i32
    %7 = "emitc.call_opaque"(%3) <{callee = "__riscv_vsetvl_e8mf4"}> : (index) -> index
    %8 = "emitc.subscript"(%0, %4) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
    %9 = "emitc.apply"(%8) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
    %10 = "emitc.call_opaque"(%9, %7) <{callee = "__riscv_vle8_v_i8mf4"}> : (!emitc.ptr<i8>, index) -> !emitc.opaque<"vint8mf4_t">
    %11 = "emitc.subscript"(%1, %4) : (!emitc.ptr<i8>, index) -> !emitc.lvalue<i8>
    %12 = "emitc.apply"(%11) {applicableOperator = "&"} : (!emitc.lvalue<i8>) -> !emitc.ptr<i8>
    %13 = "emitc.call_opaque"(%12, %7) <{callee = "__riscv_vle8_v_i8mf4"}> : (!emitc.ptr<i8>, index) -> !emitc.opaque<"vint8mf4_t">
    %14 = "emitc.call_opaque"(%5, %7) <{callee = "__riscv_vmv_v_x_i16mf2"}> : (i16, index) -> !emitc.opaque<"vint16mf2_t">
    %15 = "emitc.call_opaque"(%14, %10, %13, %7) <{callee = "__riscv_vwmacc_vv_i16mf2"}> : (!emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"vint8mf4_t">, index) -> !emitc.opaque<"vint16mf2_t">
    %16 = "emitc.call_opaque"(%6, %7) <{callee = "__riscv_vmv_v_x_i32m1"}> : (i32, index) -> !emitc.opaque<"vint32m1_t">
    %17 = "emitc.call_opaque"(%16, %15, %7) <{callee = "__riscv_vwadd_wv_i32m1"}> : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint16mf2_t">, index) -> !emitc.opaque<"vint32m1_t">
    %18 = "emitc.subscript"(%2, %4) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    %19 = "emitc.apply"(%18) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    "emitc.call_opaque"(%19, %17, %7) <{callee = "__riscv_vse32_v_i32m1"}> : (!emitc.ptr<i32>, !emitc.opaque<"vint32m1_t">, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<i8>, !emitc.ptr<i8>, !emitc.ptr<i32>, index) -> i32} : () -> ()
}) : () -> ()