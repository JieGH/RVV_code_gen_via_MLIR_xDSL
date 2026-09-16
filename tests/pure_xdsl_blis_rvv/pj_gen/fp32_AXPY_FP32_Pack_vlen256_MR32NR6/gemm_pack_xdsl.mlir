"builtin.module"() ({
  "func.func"() <{sym_name = "pack_CB_v_if_AXPY_FP32_Pack_fp32_NR6", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 6 : index} : () -> index
    "emitc.for"(%5, %3, %8) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %8) <{callee = "std::min"}> : (index, index) -> index
      %12 = "emitc.mul"(%4, %7) : (index, index) -> index
      "emitc.for"(%5, %2, %6) ({
      ^bb2(%13 : index):
        %14 = "emitc.mul"(%9, %4) : (index, index) -> index
        %15 = "emitc.add"(%14, %13) : (index, index) -> index
        %16 = "emitc.mul"(%9, %2) : (index, index) -> index
        %17 = "emitc.mul"(%13, %8) : (index, index) -> index
        %18 = "emitc.add"(%16, %17) : (index, index) -> index
        %19 = "emitc.constant"() {value = 0 : index} : () -> index
        %20 = "emitc.constant"() {value = 8 : index} : () -> index
        %21 = "emitc.call_opaque"(%11, %20) <{callee = "std::min"}> : (index, index) -> index
        %22 = "emitc.call_opaque"(%21, %19) <{callee = "std::max"}> : (index, index) -> index
        %23 = "emitc.sub"(%22, %19) : (index, index) -> index
        %24 = "emitc.call_opaque"(%23) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %25 = "emitc.mul"(%19, %4) : (index, index) -> index
        %26 = "emitc.add"(%15, %25) : (index, index) -> index
        %27 = "emitc.subscript"(%0, %26) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %28 = "emitc.apply"(%27) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %29 = "emitc.call_opaque"(%28, %12, %24) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %30 = "emitc.add"(%18, %19) : (index, index) -> index
        %31 = "emitc.subscript"(%1, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %32 = "emitc.apply"(%31) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%32, %29, %24) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_AXPY_FP32_Pack_fp32_NR6", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 6 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%8 : index):
      %9 = "emitc.sub"(%3, %8) : (index, index) -> index
      %10 = "emitc.call_opaque"(%9, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %6) ({
      ^bb2(%11 : index):
        %12 = "emitc.mul"(%11, %4) : (index, index) -> index
        %13 = "emitc.add"(%12, %8) : (index, index) -> index
        %14 = "emitc.mul"(%8, %2) : (index, index) -> index
        %15 = "emitc.mul"(%11, %7) : (index, index) -> index
        %16 = "emitc.add"(%14, %15) : (index, index) -> index
        %17 = "emitc.constant"() {value = 0 : index} : () -> index
        %18 = "emitc.constant"() {value = 8 : index} : () -> index
        %19 = "emitc.call_opaque"(%10, %18) <{callee = "std::min"}> : (index, index) -> index
        %20 = "emitc.call_opaque"(%19, %17) <{callee = "std::max"}> : (index, index) -> index
        %21 = "emitc.sub"(%20, %17) : (index, index) -> index
        %22 = "emitc.call_opaque"(%21) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %23 = "emitc.add"(%13, %17) : (index, index) -> index
        %24 = "emitc.subscript"(%0, %23) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %25 = "emitc.apply"(%24) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %26 = "emitc.call_opaque"(%25, %22) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %27 = "emitc.add"(%16, %17) : (index, index) -> index
        %28 = "emitc.subscript"(%1, %27) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %29 = "emitc.apply"(%28) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%29, %26, %22) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_AXPY_FP32_Pack_fp32_MR6", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 6 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%8 : index):
      %9 = "emitc.sub"(%2, %8) : (index, index) -> index
      %10 = "emitc.call_opaque"(%9, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %6) ({
      ^bb2(%11 : index):
        %12 = "emitc.mul"(%11, %4) : (index, index) -> index
        %13 = "emitc.add"(%12, %8) : (index, index) -> index
        %14 = "emitc.mul"(%8, %3) : (index, index) -> index
        %15 = "emitc.mul"(%11, %7) : (index, index) -> index
        %16 = "emitc.add"(%14, %15) : (index, index) -> index
        %17 = "emitc.constant"() {value = 0 : index} : () -> index
        %18 = "emitc.constant"() {value = 8 : index} : () -> index
        %19 = "emitc.call_opaque"(%10, %18) <{callee = "std::min"}> : (index, index) -> index
        %20 = "emitc.call_opaque"(%19, %17) <{callee = "std::max"}> : (index, index) -> index
        %21 = "emitc.sub"(%20, %17) : (index, index) -> index
        %22 = "emitc.call_opaque"(%21) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %23 = "emitc.add"(%13, %17) : (index, index) -> index
        %24 = "emitc.subscript"(%0, %23) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %25 = "emitc.apply"(%24) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %26 = "emitc.call_opaque"(%25, %22) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %27 = "emitc.add"(%16, %17) : (index, index) -> index
        %28 = "emitc.subscript"(%1, %27) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %29 = "emitc.apply"(%28) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%29, %26, %22) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_AXPY_FP32_Pack_fp32_MR6", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 6 : index} : () -> index
    "emitc.for"(%5, %2, %8) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%2, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %8) <{callee = "std::min"}> : (index, index) -> index
      %12 = "emitc.mul"(%4, %7) : (index, index) -> index
      "emitc.for"(%5, %3, %6) ({
      ^bb2(%13 : index):
        %14 = "emitc.mul"(%9, %4) : (index, index) -> index
        %15 = "emitc.add"(%14, %13) : (index, index) -> index
        %16 = "emitc.mul"(%9, %3) : (index, index) -> index
        %17 = "emitc.mul"(%13, %8) : (index, index) -> index
        %18 = "emitc.add"(%16, %17) : (index, index) -> index
        %19 = "emitc.constant"() {value = 0 : index} : () -> index
        %20 = "emitc.constant"() {value = 8 : index} : () -> index
        %21 = "emitc.call_opaque"(%11, %20) <{callee = "std::min"}> : (index, index) -> index
        %22 = "emitc.call_opaque"(%21, %19) <{callee = "std::max"}> : (index, index) -> index
        %23 = "emitc.sub"(%22, %19) : (index, index) -> index
        %24 = "emitc.call_opaque"(%23) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %25 = "emitc.mul"(%19, %4) : (index, index) -> index
        %26 = "emitc.add"(%15, %25) : (index, index) -> index
        %27 = "emitc.subscript"(%0, %26) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %28 = "emitc.apply"(%27) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %29 = "emitc.call_opaque"(%28, %12, %24) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %30 = "emitc.add"(%18, %19) : (index, index) -> index
        %31 = "emitc.subscript"(%1, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %32 = "emitc.apply"(%31) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%32, %29, %24) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_if_AXPY_FP32_Pack_fp32_NR32", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 32 : index} : () -> index
    "emitc.for"(%5, %3, %8) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %8) <{callee = "std::min"}> : (index, index) -> index
      %12 = "emitc.mul"(%4, %7) : (index, index) -> index
      "emitc.for"(%5, %2, %6) ({
      ^bb2(%13 : index):
        %14 = "emitc.mul"(%9, %4) : (index, index) -> index
        %15 = "emitc.add"(%14, %13) : (index, index) -> index
        %16 = "emitc.mul"(%9, %2) : (index, index) -> index
        %17 = "emitc.mul"(%13, %8) : (index, index) -> index
        %18 = "emitc.add"(%16, %17) : (index, index) -> index
        %19 = "emitc.constant"() {value = 0 : index} : () -> index
        %20 = "emitc.constant"() {value = 8 : index} : () -> index
        %21 = "emitc.call_opaque"(%11, %20) <{callee = "std::min"}> : (index, index) -> index
        %22 = "emitc.call_opaque"(%21, %19) <{callee = "std::max"}> : (index, index) -> index
        %23 = "emitc.sub"(%22, %19) : (index, index) -> index
        %24 = "emitc.call_opaque"(%23) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %25 = "emitc.mul"(%19, %4) : (index, index) -> index
        %26 = "emitc.add"(%15, %25) : (index, index) -> index
        %27 = "emitc.subscript"(%0, %26) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %28 = "emitc.apply"(%27) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %29 = "emitc.call_opaque"(%28, %12, %24) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %30 = "emitc.add"(%18, %19) : (index, index) -> index
        %31 = "emitc.subscript"(%1, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %32 = "emitc.apply"(%31) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%32, %29, %24) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %33 = "emitc.constant"() {value = 8 : index} : () -> index
        %34 = "emitc.constant"() {value = 16 : index} : () -> index
        %35 = "emitc.call_opaque"(%11, %34) <{callee = "std::min"}> : (index, index) -> index
        %36 = "emitc.call_opaque"(%35, %33) <{callee = "std::max"}> : (index, index) -> index
        %37 = "emitc.sub"(%36, %33) : (index, index) -> index
        %38 = "emitc.call_opaque"(%37) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %39 = "emitc.mul"(%33, %4) : (index, index) -> index
        %40 = "emitc.add"(%15, %39) : (index, index) -> index
        %41 = "emitc.subscript"(%0, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %42 = "emitc.apply"(%41) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %43 = "emitc.call_opaque"(%42, %12, %38) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %44 = "emitc.add"(%18, %33) : (index, index) -> index
        %45 = "emitc.subscript"(%1, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %46 = "emitc.apply"(%45) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%46, %43, %38) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %47 = "emitc.constant"() {value = 16 : index} : () -> index
        %48 = "emitc.constant"() {value = 24 : index} : () -> index
        %49 = "emitc.call_opaque"(%11, %48) <{callee = "std::min"}> : (index, index) -> index
        %50 = "emitc.call_opaque"(%49, %47) <{callee = "std::max"}> : (index, index) -> index
        %51 = "emitc.sub"(%50, %47) : (index, index) -> index
        %52 = "emitc.call_opaque"(%51) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %53 = "emitc.mul"(%47, %4) : (index, index) -> index
        %54 = "emitc.add"(%15, %53) : (index, index) -> index
        %55 = "emitc.subscript"(%0, %54) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %56 = "emitc.apply"(%55) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %57 = "emitc.call_opaque"(%56, %12, %52) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %58 = "emitc.add"(%18, %47) : (index, index) -> index
        %59 = "emitc.subscript"(%1, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %60 = "emitc.apply"(%59) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%60, %57, %52) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %61 = "emitc.constant"() {value = 24 : index} : () -> index
        %62 = "emitc.constant"() {value = 32 : index} : () -> index
        %63 = "emitc.call_opaque"(%11, %62) <{callee = "std::min"}> : (index, index) -> index
        %64 = "emitc.call_opaque"(%63, %61) <{callee = "std::max"}> : (index, index) -> index
        %65 = "emitc.sub"(%64, %61) : (index, index) -> index
        %66 = "emitc.call_opaque"(%65) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %67 = "emitc.mul"(%61, %4) : (index, index) -> index
        %68 = "emitc.add"(%15, %67) : (index, index) -> index
        %69 = "emitc.subscript"(%0, %68) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %70 = "emitc.apply"(%69) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %71 = "emitc.call_opaque"(%70, %12, %66) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %72 = "emitc.add"(%18, %61) : (index, index) -> index
        %73 = "emitc.subscript"(%1, %72) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %74 = "emitc.apply"(%73) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%74, %71, %66) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_AXPY_FP32_Pack_fp32_NR32", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 32 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%8 : index):
      %9 = "emitc.sub"(%3, %8) : (index, index) -> index
      %10 = "emitc.call_opaque"(%9, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %2, %6) ({
      ^bb2(%11 : index):
        %12 = "emitc.mul"(%11, %4) : (index, index) -> index
        %13 = "emitc.add"(%12, %8) : (index, index) -> index
        %14 = "emitc.mul"(%8, %2) : (index, index) -> index
        %15 = "emitc.mul"(%11, %7) : (index, index) -> index
        %16 = "emitc.add"(%14, %15) : (index, index) -> index
        %17 = "emitc.constant"() {value = 0 : index} : () -> index
        %18 = "emitc.constant"() {value = 8 : index} : () -> index
        %19 = "emitc.call_opaque"(%10, %18) <{callee = "std::min"}> : (index, index) -> index
        %20 = "emitc.call_opaque"(%19, %17) <{callee = "std::max"}> : (index, index) -> index
        %21 = "emitc.sub"(%20, %17) : (index, index) -> index
        %22 = "emitc.call_opaque"(%21) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %23 = "emitc.add"(%13, %17) : (index, index) -> index
        %24 = "emitc.subscript"(%0, %23) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %25 = "emitc.apply"(%24) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %26 = "emitc.call_opaque"(%25, %22) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %27 = "emitc.add"(%16, %17) : (index, index) -> index
        %28 = "emitc.subscript"(%1, %27) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %29 = "emitc.apply"(%28) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%29, %26, %22) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %30 = "emitc.constant"() {value = 8 : index} : () -> index
        %31 = "emitc.constant"() {value = 16 : index} : () -> index
        %32 = "emitc.call_opaque"(%10, %31) <{callee = "std::min"}> : (index, index) -> index
        %33 = "emitc.call_opaque"(%32, %30) <{callee = "std::max"}> : (index, index) -> index
        %34 = "emitc.sub"(%33, %30) : (index, index) -> index
        %35 = "emitc.call_opaque"(%34) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %36 = "emitc.add"(%13, %30) : (index, index) -> index
        %37 = "emitc.subscript"(%0, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %38 = "emitc.apply"(%37) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %39 = "emitc.call_opaque"(%38, %35) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %40 = "emitc.add"(%16, %30) : (index, index) -> index
        %41 = "emitc.subscript"(%1, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %42 = "emitc.apply"(%41) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%42, %39, %35) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %43 = "emitc.constant"() {value = 16 : index} : () -> index
        %44 = "emitc.constant"() {value = 24 : index} : () -> index
        %45 = "emitc.call_opaque"(%10, %44) <{callee = "std::min"}> : (index, index) -> index
        %46 = "emitc.call_opaque"(%45, %43) <{callee = "std::max"}> : (index, index) -> index
        %47 = "emitc.sub"(%46, %43) : (index, index) -> index
        %48 = "emitc.call_opaque"(%47) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %49 = "emitc.add"(%13, %43) : (index, index) -> index
        %50 = "emitc.subscript"(%0, %49) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %51 = "emitc.apply"(%50) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %52 = "emitc.call_opaque"(%51, %48) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %53 = "emitc.add"(%16, %43) : (index, index) -> index
        %54 = "emitc.subscript"(%1, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %55 = "emitc.apply"(%54) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%55, %52, %48) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %56 = "emitc.constant"() {value = 24 : index} : () -> index
        %57 = "emitc.constant"() {value = 32 : index} : () -> index
        %58 = "emitc.call_opaque"(%10, %57) <{callee = "std::min"}> : (index, index) -> index
        %59 = "emitc.call_opaque"(%58, %56) <{callee = "std::max"}> : (index, index) -> index
        %60 = "emitc.sub"(%59, %56) : (index, index) -> index
        %61 = "emitc.call_opaque"(%60) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %62 = "emitc.add"(%13, %56) : (index, index) -> index
        %63 = "emitc.subscript"(%0, %62) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %64 = "emitc.apply"(%63) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %65 = "emitc.call_opaque"(%64, %61) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %66 = "emitc.add"(%16, %56) : (index, index) -> index
        %67 = "emitc.subscript"(%1, %66) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %68 = "emitc.apply"(%67) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%68, %65, %61) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_AXPY_FP32_Pack_fp32_MR32", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 32 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%8 : index):
      %9 = "emitc.sub"(%2, %8) : (index, index) -> index
      %10 = "emitc.call_opaque"(%9, %7) <{callee = "std::min"}> : (index, index) -> index
      "emitc.for"(%5, %3, %6) ({
      ^bb2(%11 : index):
        %12 = "emitc.mul"(%11, %4) : (index, index) -> index
        %13 = "emitc.add"(%12, %8) : (index, index) -> index
        %14 = "emitc.mul"(%8, %3) : (index, index) -> index
        %15 = "emitc.mul"(%11, %7) : (index, index) -> index
        %16 = "emitc.add"(%14, %15) : (index, index) -> index
        %17 = "emitc.constant"() {value = 0 : index} : () -> index
        %18 = "emitc.constant"() {value = 8 : index} : () -> index
        %19 = "emitc.call_opaque"(%10, %18) <{callee = "std::min"}> : (index, index) -> index
        %20 = "emitc.call_opaque"(%19, %17) <{callee = "std::max"}> : (index, index) -> index
        %21 = "emitc.sub"(%20, %17) : (index, index) -> index
        %22 = "emitc.call_opaque"(%21) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %23 = "emitc.add"(%13, %17) : (index, index) -> index
        %24 = "emitc.subscript"(%0, %23) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %25 = "emitc.apply"(%24) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %26 = "emitc.call_opaque"(%25, %22) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %27 = "emitc.add"(%16, %17) : (index, index) -> index
        %28 = "emitc.subscript"(%1, %27) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %29 = "emitc.apply"(%28) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%29, %26, %22) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %30 = "emitc.constant"() {value = 8 : index} : () -> index
        %31 = "emitc.constant"() {value = 16 : index} : () -> index
        %32 = "emitc.call_opaque"(%10, %31) <{callee = "std::min"}> : (index, index) -> index
        %33 = "emitc.call_opaque"(%32, %30) <{callee = "std::max"}> : (index, index) -> index
        %34 = "emitc.sub"(%33, %30) : (index, index) -> index
        %35 = "emitc.call_opaque"(%34) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %36 = "emitc.add"(%13, %30) : (index, index) -> index
        %37 = "emitc.subscript"(%0, %36) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %38 = "emitc.apply"(%37) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %39 = "emitc.call_opaque"(%38, %35) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %40 = "emitc.add"(%16, %30) : (index, index) -> index
        %41 = "emitc.subscript"(%1, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %42 = "emitc.apply"(%41) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%42, %39, %35) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %43 = "emitc.constant"() {value = 16 : index} : () -> index
        %44 = "emitc.constant"() {value = 24 : index} : () -> index
        %45 = "emitc.call_opaque"(%10, %44) <{callee = "std::min"}> : (index, index) -> index
        %46 = "emitc.call_opaque"(%45, %43) <{callee = "std::max"}> : (index, index) -> index
        %47 = "emitc.sub"(%46, %43) : (index, index) -> index
        %48 = "emitc.call_opaque"(%47) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %49 = "emitc.add"(%13, %43) : (index, index) -> index
        %50 = "emitc.subscript"(%0, %49) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %51 = "emitc.apply"(%50) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %52 = "emitc.call_opaque"(%51, %48) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %53 = "emitc.add"(%16, %43) : (index, index) -> index
        %54 = "emitc.subscript"(%1, %53) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %55 = "emitc.apply"(%54) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%55, %52, %48) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %56 = "emitc.constant"() {value = 24 : index} : () -> index
        %57 = "emitc.constant"() {value = 32 : index} : () -> index
        %58 = "emitc.call_opaque"(%10, %57) <{callee = "std::min"}> : (index, index) -> index
        %59 = "emitc.call_opaque"(%58, %56) <{callee = "std::max"}> : (index, index) -> index
        %60 = "emitc.sub"(%59, %56) : (index, index) -> index
        %61 = "emitc.call_opaque"(%60) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %62 = "emitc.add"(%13, %56) : (index, index) -> index
        %63 = "emitc.subscript"(%0, %62) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %64 = "emitc.apply"(%63) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %65 = "emitc.call_opaque"(%64, %61) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %66 = "emitc.add"(%16, %56) : (index, index) -> index
        %67 = "emitc.subscript"(%1, %66) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %68 = "emitc.apply"(%67) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%68, %65, %61) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_AXPY_FP32_Pack_fp32_MR32", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 32 : index} : () -> index
    "emitc.for"(%5, %2, %8) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%2, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %8) <{callee = "std::min"}> : (index, index) -> index
      %12 = "emitc.mul"(%4, %7) : (index, index) -> index
      "emitc.for"(%5, %3, %6) ({
      ^bb2(%13 : index):
        %14 = "emitc.mul"(%9, %4) : (index, index) -> index
        %15 = "emitc.add"(%14, %13) : (index, index) -> index
        %16 = "emitc.mul"(%9, %3) : (index, index) -> index
        %17 = "emitc.mul"(%13, %8) : (index, index) -> index
        %18 = "emitc.add"(%16, %17) : (index, index) -> index
        %19 = "emitc.constant"() {value = 0 : index} : () -> index
        %20 = "emitc.constant"() {value = 8 : index} : () -> index
        %21 = "emitc.call_opaque"(%11, %20) <{callee = "std::min"}> : (index, index) -> index
        %22 = "emitc.call_opaque"(%21, %19) <{callee = "std::max"}> : (index, index) -> index
        %23 = "emitc.sub"(%22, %19) : (index, index) -> index
        %24 = "emitc.call_opaque"(%23) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %25 = "emitc.mul"(%19, %4) : (index, index) -> index
        %26 = "emitc.add"(%15, %25) : (index, index) -> index
        %27 = "emitc.subscript"(%0, %26) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %28 = "emitc.apply"(%27) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %29 = "emitc.call_opaque"(%28, %12, %24) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %30 = "emitc.add"(%18, %19) : (index, index) -> index
        %31 = "emitc.subscript"(%1, %30) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %32 = "emitc.apply"(%31) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%32, %29, %24) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %33 = "emitc.constant"() {value = 8 : index} : () -> index
        %34 = "emitc.constant"() {value = 16 : index} : () -> index
        %35 = "emitc.call_opaque"(%11, %34) <{callee = "std::min"}> : (index, index) -> index
        %36 = "emitc.call_opaque"(%35, %33) <{callee = "std::max"}> : (index, index) -> index
        %37 = "emitc.sub"(%36, %33) : (index, index) -> index
        %38 = "emitc.call_opaque"(%37) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %39 = "emitc.mul"(%33, %4) : (index, index) -> index
        %40 = "emitc.add"(%15, %39) : (index, index) -> index
        %41 = "emitc.subscript"(%0, %40) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %42 = "emitc.apply"(%41) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %43 = "emitc.call_opaque"(%42, %12, %38) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %44 = "emitc.add"(%18, %33) : (index, index) -> index
        %45 = "emitc.subscript"(%1, %44) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %46 = "emitc.apply"(%45) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%46, %43, %38) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %47 = "emitc.constant"() {value = 16 : index} : () -> index
        %48 = "emitc.constant"() {value = 24 : index} : () -> index
        %49 = "emitc.call_opaque"(%11, %48) <{callee = "std::min"}> : (index, index) -> index
        %50 = "emitc.call_opaque"(%49, %47) <{callee = "std::max"}> : (index, index) -> index
        %51 = "emitc.sub"(%50, %47) : (index, index) -> index
        %52 = "emitc.call_opaque"(%51) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %53 = "emitc.mul"(%47, %4) : (index, index) -> index
        %54 = "emitc.add"(%15, %53) : (index, index) -> index
        %55 = "emitc.subscript"(%0, %54) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %56 = "emitc.apply"(%55) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %57 = "emitc.call_opaque"(%56, %12, %52) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %58 = "emitc.add"(%18, %47) : (index, index) -> index
        %59 = "emitc.subscript"(%1, %58) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %60 = "emitc.apply"(%59) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%60, %57, %52) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        %61 = "emitc.constant"() {value = 24 : index} : () -> index
        %62 = "emitc.constant"() {value = 32 : index} : () -> index
        %63 = "emitc.call_opaque"(%11, %62) <{callee = "std::min"}> : (index, index) -> index
        %64 = "emitc.call_opaque"(%63, %61) <{callee = "std::max"}> : (index, index) -> index
        %65 = "emitc.sub"(%64, %61) : (index, index) -> index
        %66 = "emitc.call_opaque"(%65) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
        %67 = "emitc.mul"(%61, %4) : (index, index) -> index
        %68 = "emitc.add"(%15, %67) : (index, index) -> index
        %69 = "emitc.subscript"(%0, %68) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %70 = "emitc.apply"(%69) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %71 = "emitc.call_opaque"(%70, %12, %66) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %72 = "emitc.add"(%18, %61) : (index, index) -> index
        %73 = "emitc.subscript"(%1, %72) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %74 = "emitc.apply"(%73) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%74, %71, %66) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
}) : () -> ()
