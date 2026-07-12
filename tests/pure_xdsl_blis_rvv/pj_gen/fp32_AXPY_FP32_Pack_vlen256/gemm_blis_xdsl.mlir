"builtin.module"() ({
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR8_NR8", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR8_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR8_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 8 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 8 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 8 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 8 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR8_NR8"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR8_NR7", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR8_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR8_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 7 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 7 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 8 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 8 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR8_NR7"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR8_NR6", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR8_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR8_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 6 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 6 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 8 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 8 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR8_NR6"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR8_NR5", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR8_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR8_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 5 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 5 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 8 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 8 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR8_NR5"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR8_NR4", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR8_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR8_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 4 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 4 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 8 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 8 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR8_NR4"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR8_NR3", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR8_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR8_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 3 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 3 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 8 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 8 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR8_NR3"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR8_NR2", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR8_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR8_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 2 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 2 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 8 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 8 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR8_NR2"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR7_NR8", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR7_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR7_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 8 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 8 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 7 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 7 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR7_NR8"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR7_NR7", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR7_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR7_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 7 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 7 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 7 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 7 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR7_NR7"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR7_NR6", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR7_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR7_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 6 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 6 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 7 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 7 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR7_NR6"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR7_NR5", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR7_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR7_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 5 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 5 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 7 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 7 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR7_NR5"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR7_NR4", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR7_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR7_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 4 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 4 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 7 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 7 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR7_NR4"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR7_NR3", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR7_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR7_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 3 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 3 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 7 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 7 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR7_NR3"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR7_NR2", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR7_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR7_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 2 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 2 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 7 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 7 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR7_NR2"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR6_NR8", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR6_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR6_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 8 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 8 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 6 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 6 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR6_NR8"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR6_NR7", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR6_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR6_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 7 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 7 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 6 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 6 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR6_NR7"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR6_NR6", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR6_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR6_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 6 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 6 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 6 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 6 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR6_NR6"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR6_NR5", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR6_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR6_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 5 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 5 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 6 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 6 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR6_NR5"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR6_NR4", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR6_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR6_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 4 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 4 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 6 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 6 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR6_NR4"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR6_NR3", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR6_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR6_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 3 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 3 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 6 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 6 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR6_NR3"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR6_NR2", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR6_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR6_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 2 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 2 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 6 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 6 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR6_NR2"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR5_NR8", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR5_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR5_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 8 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 8 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 5 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 5 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR5_NR8"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR5_NR7", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR5_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR5_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 7 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 7 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 5 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 5 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR5_NR7"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR5_NR6", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR5_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR5_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 6 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 6 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 5 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 5 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR5_NR6"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR5_NR5", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR5_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR5_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 5 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 5 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 5 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 5 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR5_NR5"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR5_NR4", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR5_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR5_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 4 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 4 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 5 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 5 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR5_NR4"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR5_NR3", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR5_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR5_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 3 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 3 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 5 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 5 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR5_NR3"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR5_NR2", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR5_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR5_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 2 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 2 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 5 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 5 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR5_NR2"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR4_NR8", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR4_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR4_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 8 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 8 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 4 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 4 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR4_NR8"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR4_NR7", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR4_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR4_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 7 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 7 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 4 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 4 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR4_NR7"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR4_NR6", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR4_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR4_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 6 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 6 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 4 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 4 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR4_NR6"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR4_NR5", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR4_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR4_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 5 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 5 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 4 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 4 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR4_NR5"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR4_NR4", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR4_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR4_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 4 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 4 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 4 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 4 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR4_NR4"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR4_NR3", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR4_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR4_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 3 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 3 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 4 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 4 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR4_NR3"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR4_NR2", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR4_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR4_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 2 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 2 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 4 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 4 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR4_NR2"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR3_NR8", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR3_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR3_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 8 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 8 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 3 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 3 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR3_NR8"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR3_NR7", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR3_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR3_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 7 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 7 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 3 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 3 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR3_NR7"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR3_NR6", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR3_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR3_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 6 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 6 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 3 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 3 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR3_NR6"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR3_NR5", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR3_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR3_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 5 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 5 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 3 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 3 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR3_NR5"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR3_NR4", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR3_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR3_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 4 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 4 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 3 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 3 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR3_NR4"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR3_NR3", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR3_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR3_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 3 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 3 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 3 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 3 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR3_NR3"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR3_NR2", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR3_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR3_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 2 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 2 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 3 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 3 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR3_NR2"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR2_NR8", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR2_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR2_NR8"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 8 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 8 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 2 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 2 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR2_NR8"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR2_NR7", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR2_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR2_NR7"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 7 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 7 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 2 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 2 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR2_NR7"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR2_NR6", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR2_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR2_NR6"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 6 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 6 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 2 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 2 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR2_NR6"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR2_NR5", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR2_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR2_NR5"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 5 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 5 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 2 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 2 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR2_NR5"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR2_NR4", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR2_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR2_NR4"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 4 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 4 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 2 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 2 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR2_NR4"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR2_NR3", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR2_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR2_NR3"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 3 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 3 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 2 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 2 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR2_NR3"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
  "func.func"() <{sym_name = "gemm_blis_v_fp32_MR2_NR2", function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()}> ({
  ^bb0(%0 : i32, %1 : i32, %2 : i32, %3 : f32, %4 : !emitc.ptr<f32>, %5 : i32, %6 : !emitc.ptr<f32>, %7 : i32, %8 : f32, %9 : !emitc.ptr<f32>, %10 : i32, %11 : !emitc.ptr<f32>, %12 : !emitc.ptr<f32>, %13 : i32, %14 : i32, %15 : i32):
    %16 = "emitc.constant"() {value = 0 : i32} : () -> i32
    "emitc.for"(%16, %1, %14) ({
    ^bb1(%17 : i32):
      %18 = "emitc.sub"(%1, %17) : (i32, i32) -> i32
      %19 = "emitc.call_opaque"(%18, %14) <{callee = "std::min"}> : (i32, i32) -> i32
      "emitc.for"(%16, %2, %15) ({
      ^bb2(%20 : i32):
        %21 = "emitc.sub"(%2, %20) : (i32, i32) -> i32
        %22 = "emitc.call_opaque"(%21, %15) <{callee = "std::min"}> : (i32, i32) -> i32
        %23 = "emitc.mul"(%17, %7) : (i32, i32) -> i32
        %24 = "emitc.add"(%23, %20) : (i32, i32) -> i32
        %25 = "emitc.call_opaque"(%6, %24) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        "emitc.call_opaque"(%25, %12, %22, %19, %7) <{callee = "pack_CB_v_if_fp32_MR2_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
        %26 = "emitc.call_opaque"(%20, %8) <{callee = "GET_BETA"}> : (i32, f32) -> f32
        "emitc.for"(%16, %0, %13) ({
        ^bb3(%27 : i32):
          %28 = "emitc.sub"(%0, %27) : (i32, i32) -> i32
          %29 = "emitc.call_opaque"(%28, %13) <{callee = "std::min"}> : (i32, i32) -> i32
          %30 = "emitc.mul"(%20, %5) : (i32, i32) -> i32
          %31 = "emitc.add"(%30, %27) : (i32, i32) -> i32
          %32 = "emitc.call_opaque"(%4, %31) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
          "emitc.call_opaque"(%32, %11, %29, %22, %5) <{callee = "pack_RB_v_if_fp32_MR2_NR2"}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> ()
          %33 = "emitc.constant"() {value = 2 : i32} : () -> i32
          "emitc.for"(%16, %19, %33) ({
          ^bb4(%34 : i32):
            %35 = "emitc.sub"(%19, %34) : (i32, i32) -> i32
            %36 = "emitc.constant"() {value = 2 : i32} : () -> i32
            %37 = "emitc.call_opaque"(%35, %36) <{callee = "std::min"}> : (i32, i32) -> i32
            %38 = "emitc.constant"() {value = 2 : i32} : () -> i32
            "emitc.for"(%16, %29, %38) ({
            ^bb5(%39 : i32):
              %40 = "emitc.sub"(%29, %39) : (i32, i32) -> i32
              %41 = "emitc.constant"() {value = 2 : i32} : () -> i32
              %42 = "emitc.call_opaque"(%40, %41) <{callee = "std::min"}> : (i32, i32) -> i32
              %43 = "emitc.add"(%27, %39) : (i32, i32) -> i32
              %44 = "emitc.add"(%17, %34) : (i32, i32) -> i32
              %45 = "emitc.mul"(%44, %10) : (i32, i32) -> i32
              %46 = "emitc.add"(%45, %43) : (i32, i32) -> i32
              %47 = "emitc.call_opaque"(%9, %46) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %48 = "emitc.mul"(%39, %22) : (i32, i32) -> i32
              %49 = "emitc.call_opaque"(%11, %48) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              %50 = "emitc.mul"(%34, %22) : (i32, i32) -> i32
              %51 = "emitc.call_opaque"(%12, %50) <{callee = "GET_PTR"}> : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
              "emitc.call_opaque"(%42, %37, %22, %3, %49, %51, %26, %47, %10) <{callee = "gemm_microkernel_v_dispatch_fp32_MR2_NR2"}> : (i32, i32, i32, f32, !emitc.ptr<f32>, !emitc.ptr<f32>, f32, !emitc.ptr<f32>, i32) -> ()
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
  }) {function_type = (i32, i32, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, i32, f32, !emitc.ptr<f32>, i32, !emitc.ptr<f32>, !emitc.ptr<f32>, i32, i32, i32) -> i32} : () -> ()
}) : () -> ()