import subprocess

from xdsl.context import Context
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    FunctionType,
    IndexType,
    IntegerType,
    MemRefType,
    ModuleOp,
)
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from xdsltemplate.dialects.rvv import (
    SetvlMf4Op,
    vle8_v_i8mf4Op,
    vmv_v_x_i16mf2Op,
    vmv_v_x_i32m1Op,
    vse32_v_i32m1Op,
    vwadd_wv_i32m1Op,
    vwmacc_vv_i16mf2Op,
)
from xdsltemplate.transforms.arith_to_emitc import ArithToEmitCPass
from xdsltemplate.transforms.memref_to_emitc import MemRefToEmitCPass
from xdsltemplate.transforms.rvv_to_emitc import RVVToEmitCPass
from xdsltemplate.transforms.scf_to_emitc import SCFToEmitCPass


def build_mixed_precision_test() -> ModuleOp:
    """
    Builds an MLIR module representing mixed precision math using custom RVV intrinsics.
    Signature: void test_mixed_precision(A: memref<i8>, B: memref<i8>, C: memref<i32>, M: index)
    """
    index_t = IndexType()
    i8_t = IntegerType(8)
    i32_t = IntegerType(32)

    memref_i8 = MemRefType(i8_t, [-1])
    memref_i32 = MemRefType(i32_t, [-1])

    input_types = [
        memref_i8,  # A
        memref_i8,  # B
        memref_i32,  # C
        index_t,  # M (size)
    ]

    func_type = FunctionType.from_lists(input_types, [])
    entry_block = Block(arg_types=input_types)
    test_func = func.FuncOp("test_mixed_precision", func_type, Region([entry_block]))

    A, B, C, M_arg = entry_block.args

    # Constants
    op_zero = arith.ConstantOp.from_int_and_width(0, index_t)
    entry_block.add_op(op_zero)
    zero = op_zero.result

    op_zero_i16 = arith.ConstantOp.from_int_and_width(0, IntegerType(16))
    entry_block.add_op(op_zero_i16)
    zero_i16 = op_zero_i16.result

    op_zero_i32 = arith.ConstantOp.from_int_and_width(0, i32_t)
    entry_block.add_op(op_zero_i32)
    zero_i32 = op_zero_i32.result

    # Set vl using correct e8mf4 setvl for vint8mf4_t/vint16mf2_t/vint32m1_t
    setvl = SetvlMf4Op(M_arg)
    entry_block.add_op(setvl)
    vl = setvl.vl

    # Load A as vint8mf4_t
    load_a = vle8_v_i8mf4Op(A, zero, vl)
    entry_block.add_op(load_a)

    # Load B as vint8mf4_t
    load_b = vle8_v_i8mf4Op(B, zero, vl)
    entry_block.add_op(load_b)

    # Zero-init i16 accumulator (vint16mf2_t)
    acc_i16 = vmv_v_x_i16mf2Op(zero_i16, vl)
    entry_block.add_op(acc_i16)

    # vwmacc: acc_i16 += A_i8 * B_i8  (i8 * i8 -> widened into i16)
    # Replaces the separate vwmul + vwadd.wv sequence
    mac = vwmacc_vv_i16mf2Op(acc_i16.result, load_a.result, load_b.result, vl)
    entry_block.add_op(mac)

    # Widen i16 -> i32: zero_i32 + i16_result -> vint32m1_t
    zero_acc_i32 = vmv_v_x_i32m1Op(zero_i32, vl)
    entry_block.add_op(zero_acc_i32)

    sum_i32 = vwadd_wv_i32m1Op(zero_acc_i32.result, mac.result, vl)
    entry_block.add_op(sum_i32)

    # Store vint32m1_t result to C (memref<i32>)
    store_c = vse32_v_i32m1Op(C, zero, sum_i32.result, vl)
    entry_block.add_op(store_c)

    # Return
    entry_block.add_op(func.ReturnOp())

    return ModuleOp([test_func])


def lower_to_emitc(module: ModuleOp, ctx: Context):
    MemRefToEmitCPass().apply(ctx, module)
    ArithToEmitCPass().apply(ctx, module)
    SCFToEmitCPass().apply(ctx, module)
    RVVToEmitCPass().apply(ctx, module)
    return module


if __name__ == "__main__":
    ctx = Context()
    module = build_mixed_precision_test()

    print("--- MLIR ---")
    printer = Printer(print_generic_format=False)
    printer.print_op(module)
    print("\n")

    module = lower_to_emitc(module, ctx)

    print("--- EmitC ---")
    printer.print_op(module)
    print("\n")

    # Translate to C — write using GENERIC format (required by mlir-translate)
    with open("tmp_int_codeGen/mixed_precision.mlir", "w") as f:
        Printer(stream=f, print_generic_format=True).print_op(module)

    MLIR_TRANSLATE = "/home/jlei/Documents/ws_MLIR/llvm_pj_dir_06092025/llvm-project/build/bin/mlir-translate"
    C_OUT = "tmp_int_codeGen/mixed_precision.c"
    try:
        result = subprocess.run(
            [
                MLIR_TRANSLATE,
                "-allow-unregistered-dialect",
                "-mlir-to-cpp",
                "tmp_int_codeGen/mixed_precision.mlir",
                "-o",
                C_OUT,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print("mlir-translate failed:")
            print(result.stderr)
        else:
            with open(C_OUT) as f:
                print("--- Generated C++ Code ---")
                print(f.read())
    except FileNotFoundError:
        print(f"mlir-translate not found at: {MLIR_TRANSLATE}")

    # test_mixed_precision_arith.py complete
