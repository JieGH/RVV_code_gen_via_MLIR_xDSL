"builtin.module"() ({
  "func.func"() <{sym_name = "pack_CB_v_if_fp32_RR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 8 : index} : () -> index
    "emitc.for"(%5, %3, %8) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%3, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %8) <{callee = "std::min"}> : (index, index) -> index
      %12 = "emitc.call_opaque"(%11) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
      %13 = "emitc.mul"(%4, %7) : (index, index) -> index
      "emitc.for"(%5, %2, %6) ({
      ^bb2(%14 : index):
        %15 = "emitc.mul"(%9, %4) : (index, index) -> index
        %16 = "emitc.add"(%15, %14) : (index, index) -> index
        %17 = "emitc.subscript"(%0, %16) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %18 = "emitc.apply"(%17) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %19 = "emitc.call_opaque"(%18, %13, %12) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %20 = "emitc.mul"(%9, %2) : (index, index) -> index
        %21 = "emitc.mul"(%14, %8) : (index, index) -> index
        %22 = "emitc.add"(%20, %21) : (index, index) -> index
        %23 = "emitc.subscript"(%1, %22) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %24 = "emitc.apply"(%23) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%24, %19, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_CB_v_else_fp32_RR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 8 : index} : () -> index
    "emitc.for"(%5, %3, %7) ({
    ^bb1(%8 : index):
      %9 = "emitc.sub"(%3, %8) : (index, index) -> index
      %10 = "emitc.call_opaque"(%9, %7) <{callee = "std::min"}> : (index, index) -> index
      %11 = "emitc.call_opaque"(%10) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
      "emitc.for"(%5, %2, %6) ({
      ^bb2(%12 : index):
        %13 = "emitc.mul"(%12, %4) : (index, index) -> index
        %14 = "emitc.add"(%13, %8) : (index, index) -> index
        %15 = "emitc.subscript"(%0, %14) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %16 = "emitc.apply"(%15) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %17 = "emitc.call_opaque"(%16, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %18 = "emitc.mul"(%8, %2) : (index, index) -> index
        %19 = "emitc.mul"(%12, %7) : (index, index) -> index
        %20 = "emitc.add"(%18, %19) : (index, index) -> index
        %21 = "emitc.subscript"(%1, %20) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %22 = "emitc.apply"(%21) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%22, %17, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_if_fp32_RR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 8 : index} : () -> index
    "emitc.for"(%5, %2, %7) ({
    ^bb1(%8 : index):
      %9 = "emitc.sub"(%2, %8) : (index, index) -> index
      %10 = "emitc.call_opaque"(%9, %7) <{callee = "std::min"}> : (index, index) -> index
      %11 = "emitc.call_opaque"(%10) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
      "emitc.for"(%5, %3, %6) ({
      ^bb2(%12 : index):
        %13 = "emitc.mul"(%12, %4) : (index, index) -> index
        %14 = "emitc.add"(%13, %8) : (index, index) -> index
        %15 = "emitc.subscript"(%0, %14) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %16 = "emitc.apply"(%15) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %17 = "emitc.call_opaque"(%16, %11) <{callee = "__riscv_vle32_v_f32m1"}> : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
        %18 = "emitc.mul"(%8, %3) : (index, index) -> index
        %19 = "emitc.mul"(%12, %7) : (index, index) -> index
        %20 = "emitc.add"(%18, %19) : (index, index) -> index
        %21 = "emitc.subscript"(%1, %20) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %22 = "emitc.apply"(%21) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%22, %17, %11) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "pack_RB_v_else_fp32_RR8", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> ()}> ({
  ^bb0(%0 : !emitc.ptr<f32>, %1 : !emitc.ptr<f32>, %2 : index, %3 : index, %4 : index):
    %5 = "emitc.constant"() {value = 0 : index} : () -> index
    %6 = "emitc.constant"() {value = 1 : index} : () -> index
    %7 = "emitc.constant"() {value = 4 : index} : () -> index
    %8 = "emitc.constant"() {value = 8 : index} : () -> index
    "emitc.for"(%5, %2, %8) ({
    ^bb1(%9 : index):
      %10 = "emitc.sub"(%2, %9) : (index, index) -> index
      %11 = "emitc.call_opaque"(%10, %8) <{callee = "std::min"}> : (index, index) -> index
      %12 = "emitc.call_opaque"(%11) <{callee = "__riscv_vsetvl_e32m1"}> : (index) -> index
      %13 = "emitc.mul"(%4, %7) : (index, index) -> index
      "emitc.for"(%5, %3, %6) ({
      ^bb2(%14 : index):
        %15 = "emitc.mul"(%9, %4) : (index, index) -> index
        %16 = "emitc.add"(%15, %14) : (index, index) -> index
        %17 = "emitc.subscript"(%0, %16) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %18 = "emitc.apply"(%17) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        %19 = "emitc.call_opaque"(%18, %13, %12) <{callee = "__riscv_vlse32_v_f32m1"}> : (!emitc.ptr<f32>, index, index) -> !emitc.opaque<"vfloat32m1_t">
        %20 = "emitc.mul"(%9, %3) : (index, index) -> index
        %21 = "emitc.mul"(%14, %8) : (index, index) -> index
        %22 = "emitc.add"(%20, %21) : (index, index) -> index
        %23 = "emitc.subscript"(%1, %22) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %24 = "emitc.apply"(%23) {applicableOperator = "&"} : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%24, %19, %12) <{callee = "__riscv_vse32_v_f32m1"}> : (!emitc.ptr<f32>, !emitc.opaque<"vfloat32m1_t">, index) -> ()
        "emitc.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "emitc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>, index, index, index) -> index} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_CN_CN_C_fp32", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>):
    %13 = "emitc.constant"() {value = 0 : i32} : () -> i32
    %14 = "emitc.constant"() {value = 384 : i32} : () -> i32
    "emitc.for"(%13, %1, %14) ({
    ^bb1(%15 : i32):
      %16 = "emitc.sub"(%1, %15) : (i32, i32) -> i32
      %17 = "emitc.constant"() {value = 384 : i32} : () -> i32
      %18 = "emitc.call_opaque"(%16, %17) <{callee = "std::min"}> : (i32, i32) -> i32
      %19 = "emitc.constant"() {value = 256 : i32} : () -> i32
      "emitc.for"(%13, %2, %19) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.constant"() {value = 256 : i32} : () -> i32
        %23 = "emitc.call_opaque"(%21, %22) <{callee = "std::min"}> : (i32, i32) -> i32
        %24 = "emitc.mul"(%15, %7) : (i32, i32) -> i32
        %25 = "emitc.add"(%24, %20) : (i32, i32) -> i32
        %26 = "emitc.call_opaque"(%6, %25) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%26, %12, %23, %18, %7) <{callee = "pack_CB_v_if_fp32_RR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %27 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        %28 = "emitc.constant"() {value = 384 : i32} : () -> i32
        "emitc.for"(%13, %0, %28) ({
        ^bb3(%29 : i32):
          %30 = "emitc.sub"(%0, %29) : (i32, i32) -> i32
          %31 = "emitc.constant"() {value = 384 : i32} : () -> i32
          %32 = "emitc.call_opaque"(%30, %31) <{callee = "std::min"}> : (i32, i32) -> i32
          %33 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %34 = "emitc.add"(%33, %29) : (i32, i32) -> i32
          %35 = "emitc.call_opaque"(%4, %34) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%35, %11, %32, %23, %5) <{callee = "pack_RB_v_if_fp32_RR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %36 = "emitc.constant"() {value = 8 : i32} : () -> i32
          "emitc.for"(%13, %18, %36) ({
          ^bb4(%37 : i32):
            %38 = "emitc.sub"(%18, %37) : (i32, i32) -> i32
            %39 = "emitc.constant"() {value = 8 : i32} : () -> i32
            %40 = "emitc.call_opaque"(%38, %39) <{callee = "std::min"}> : (i32, i32) -> i32
            %41 = "emitc.constant"() {value = 8 : i32} : () -> i32
            "emitc.for"(%13, %32, %41) ({
            ^bb5(%42 : i32):
              %43 = "emitc.sub"(%32, %42) : (i32, i32) -> i32
              %44 = "emitc.constant"() {value = 8 : i32} : () -> i32
              %45 = "emitc.call_opaque"(%43, %44) <{callee = "std::min"}> : (i32, i32) -> i32
              %46 = "emitc.add"(%29, %42) : (i32, i32) -> i32
              %47 = "emitc.add"(%15, %37) : (i32, i32) -> i32
              %48 = "emitc.mul"(%47, %10) : (i32, i32) -> i32
              %49 = "emitc.add"(%48, %46) : (i32, i32) -> i32
              %50 = "emitc.call_opaque"(%9, %49) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %51 = "emitc.mul"(%42, %23) : (i32, i32) -> i32
              %52 = "emitc.call_opaque"(%11, %51) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %53 = "emitc.mul"(%37, %23) : (i32, i32) -> i32
              %54 = "emitc.call_opaque"(%12, %53) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%45, %40, %23, %3, %52, %54, %27, %50, %10) <{callee = "gemm_microkernel_v"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
              "emitc.yield"() : () -> ()
            }) : (i32, i32, i32) -> ()
            "emitc.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "emitc.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "emitc.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "emitc.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>) -> i32} : () -> ()
}) : () -> ()
