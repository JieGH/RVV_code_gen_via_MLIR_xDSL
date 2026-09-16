"""
Standalone verification for INT8 RVV dialect + rvv_to_emitc lowering patterns.

Tests each new op individually and verifies the emitted C intrinsic name
matches the reference hand-written C code.

Run:
    python tests/verify_int8_rvv.py
"""

from io import StringIO

from xdsl.context import Context
from xdsl.dialects import arith, builtin, emitc, func, memref, scf
from xdsl.dialects.builtin import (
    FunctionType,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
)
from xdsl.ir import Block, Region
from xdsl.printer import Printer

# ── Dialect + Pass imports ───────────────────────────────────────────────────
from xdsltemplate.dialects.rvv import (
    RVV,
    RVVInt32M4Type,
    vle8_v_i8m1Op,
    vmv_v_x_i32m4Op,
    vse32_v_i32m4Op,
    vwadd_vx_i16m2Op,
    vwmacc_vx_i32m4Op,
)
from xdsltemplate.transforms.rvv_to_emitc import RVVToEmitCPass

# ── Reference intrinsic names (from the hand-written C) ──────────────────────
EXPECTED_INTRINSICS = [
    "__riscv_vmv_v_x_i32m4",  # zero-init accumulator
    "__riscv_vle8_v_i8m1",  # load A
    "__riscv_vwadd_vx_i16m2",  # widen A: i8 → i16
    "__riscv_vwmacc_vx_i32m4",  # widening MAC
    "__riscv_vle32_v_i32m4",  # load C (beta≠0)
    "__riscv_vse32_v_i32m4",  # store C
]


# ─────────────────────────────────────────────────────────────────────────────
def make_context() -> Context:
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(emitc.EmitC)
    ctx.load_dialect(RVV)
    return ctx


def print_module(module: ModuleOp) -> str:
    buf = StringIO()
    Printer(stream=buf).print(module)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
def build_test_module() -> ModuleOp:
    """Builds the complete 32x6 INT8 ukernel matching the provided C code."""
    index_t = IndexType()
    i8_t = IntegerType(8)
    i32_t = IntegerType(32)
    memref_i8 = MemRefType(i8_t, [-1])
    memref_i32 = MemRefType(i32_t, [-1])
    acc_t = RVVInt32M4Type()

    # void uk_riscv1_gemm_Ccol_32x6_vl256_m4_int8_int32(
    #    int mr, int nr, size_t kc, int8_t *Ar, int8_t *Br, int32_t *Cr, int32_t beta, size_t ldC)
    input_types = [
        i32_t,
        i32_t,
        index_t,  # mr, nr, kc
        memref_i8,
        memref_i8,
        memref_i32,
        i32_t,
        index_t,  # beta, ldC
    ]
    func_type = FunctionType.from_lists(input_types, [])
    entry_block = Block(arg_types=input_types)
    mr, nr, kc, Ar, Br, Cr, beta, ldC = entry_block.args

    c_vl = arith.ConstantOp(IntegerAttr(32, index_t))
    entry_block.add_op(c_vl)
    vl = c_vl.result

    c0_i32 = arith.ConstantOp(IntegerAttr(0, i32_t))
    entry_block.add_op(c0_i32)
    c0_idx = arith.ConstantOp(IntegerAttr(0, index_t))
    entry_block.add_op(c0_idx)
    c1_idx = arith.ConstantOp(IntegerAttr(1, index_t))
    entry_block.add_op(c1_idx)
    c32_idx = arith.ConstantOp(IntegerAttr(32, index_t))
    entry_block.add_op(c32_idx)
    c6_idx = arith.ConstantOp(IntegerAttr(6, index_t))
    entry_block.add_op(c6_idx)
    c0_i8 = arith.ConstantOp(IntegerAttr(0, i8_t))
    entry_block.add_op(c0_i8)

    # ----- INIT ACCUMULATORS (Beta = 0 assumed) -----
    init_accs = []
    for _ in range(6):
        vmv = vmv_v_x_i32m4Op(c0_i32.result, vl)
        entry_block.add_op(vmv)
        init_accs.append(vmv.result)

    # ----- FOR LOOP -----
    # Loop state: c0..c5, bA, bB
    acc_types = [acc_t] * 6
    loop_body_types = [index_t] + acc_types + [index_t, index_t]
    loop_body = Block(arg_types=loop_body_types)
    pr = loop_body.args[0]
    l_accs = [loop_body.args[n + 1] for n in range(6)]
    bA = loop_body.args[7]
    bB = loop_body.args[8]

    # vtmp = vle8(&Ar[bA+0]), A0 = vwadd(vtmp, 0)
    vle8 = vle8_v_i8m1Op(Ar, bA, vl)
    loop_body.add_op(vle8)
    vwadd = vwadd_vx_i16m2Op(vle8.result, c0_i8.result, vl)
    loop_body.add_op(vwadd)
    A0 = vwadd.result

    # C0x = vwmacc(C0x, Br[bB + n], A0)
    new_accs = []
    for n in range(6):
        n_idx = arith.ConstantOp(IntegerAttr(n, index_t))
        loop_body.add_op(n_idx)
        bb_off = arith.AddiOp(bB, n_idx.result)
        loop_body.add_op(bb_off)
        br_scalar = memref.LoadOp.get(Br, [bb_off.result])
        loop_body.add_op(br_scalar)
        vwm = vwmacc_vx_i32m4Op(l_accs[n], br_scalar.res, A0, vl)
        loop_body.add_op(vwm)
        new_accs.append(vwm.result)

    # bA += 32, bB += 6
    next_bA = arith.AddiOp(bA, c32_idx.result)
    loop_body.add_op(next_bA)
    next_bB = arith.AddiOp(bB, c6_idx.result)
    loop_body.add_op(next_bB)

    # yield updated
    loop_yield_vals = new_accs + [next_bA.result, next_bB.result]
    loop_body.add_op(scf.YieldOp(*loop_yield_vals))

    # ForOp Setup -> iter parameters: acc0..5, 0 (bA), 0 (bB)
    iter_inits = init_accs + [c0_idx.result, c0_idx.result]
    for_op = scf.ForOp(
        c0_idx.result, kc, c1_idx.result, iter_inits, Region([loop_body])
    )
    entry_block.add_op(for_op)

    # ----- STORE -----
    final_accs = [for_op.results[n] for n in range(6)]
    for n in range(6):
        n_idx = arith.ConstantOp(IntegerAttr(n, index_t))
        entry_block.add_op(n_idx)
        col_offset = arith.MuliOp(n_idx.result, ldC)
        entry_block.add_op(col_offset)
        vse = vse32_v_i32m4Op(Cr, col_offset.result, final_accs[n], vl)
        entry_block.add_op(vse)

    entry_block.add_op(func.ReturnOp())

    gemm_func = func.FuncOp(
        "uk_riscv1_gemm_Ccol_32x6_vl256_m4_int8_int32", func_type, Region([entry_block])
    )
    return ModuleOp([gemm_func])


# ─────────────────────────────────────────────────────────────────────────────
def verify_emitc_output(emitc_str: str):
    """Check that every expected intrinsic appears in the lowered EmitC output."""
    print("\n── EmitC output ──────────────────────────────────────────")
    print(emitc_str)
    print("──────────────────────────────────────────────────────────\n")

    all_ok = True
    for intrinsic in EXPECTED_INTRINSICS:
        if intrinsic in emitc_str:
            print(f"  ✅  {intrinsic}")
        else:
            print(f"  ❌  MISSING: {intrinsic}")
            all_ok = False

    print()
    if all_ok:
        print("All intrinsics verified ✅  — INT8 RVV dialect + lowering OK")
    else:
        print("FAILURES detected ❌  — check pattern lowering in rvv_to_emitc.py")
    return all_ok


# Path to mlir-translate binary
MLIR_TRANSLATE = "/home/jlei/Documents/ws_MLIR/llvm-project/build/bin/mlir-translate"
OUT_DIR = "/home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/RVV_code_gen_via_MLIR_xDSL/tests/tmp_int_codeGen"
MLIR_TMP = f"{OUT_DIR}/verify_int8_rvv.mlir"
C_TMP = f"{OUT_DIR}/verify_int8_rvv.c"


# ─────────────────────────────────────────────────────────────────────────────
def translate_to_c(module: ModuleOp) -> str | None:
    """Write Generic EmitC MLIR to OUT_DIR and call mlir-translate -mlir-to-cpp."""
    import os
    import subprocess

    os.makedirs(OUT_DIR, exist_ok=True)

    # Needs to be generic format for mlir-translate to parse
    buf = StringIO()
    Printer(stream=buf, print_generic_format=True).print_op(module)
    emitc_mlir_generic = buf.getvalue().replace("<-1x", "<?x")

    with open(MLIR_TMP, "w") as f:
        f.write(emitc_mlir_generic)
    try:
        result = subprocess.run(
            [
                MLIR_TRANSLATE,
                "-allow-unregistered-dialect",
                "-mlir-to-cpp",
                MLIR_TMP,
                "-o",
                C_TMP,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"mlir-translate error:\n{result.stderr}")
            return None
        with open(C_TMP) as f:
            return f.read()
    except FileNotFoundError:
        print(f"mlir-translate not found at: {MLIR_TRANSLATE}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
from xdsltemplate.transforms.arith_to_emitc import ArithToEmitCPass
from xdsltemplate.transforms.memref_load_to_emitc import MemrefLoadToEmitcPass
from xdsltemplate.transforms.memref_store_to_emitc import MemrefStoreToEmitcPass
from xdsltemplate.transforms.memref_to_emitc import MemRefToEmitCPass
from xdsltemplate.transforms.scf_to_emitc import SCFToEmitCPass


def main():
    print("=== INT8 RVV Dialect + EmitC Lowering Verification ===\n")

    # 1. Build the test MLIR module
    module = build_test_module()
    print("── [1] MLIR before lowering ──────────────────────────────")
    print(print_module(module))

    # 2. Apply the full emitc pass pipeline
    ctx = make_context()
    print("  Applying MemRefToEmitCPass...", flush=True)
    MemRefToEmitCPass().apply(ctx, module)
    print("  Applying ArithToEmitCPass...", flush=True)
    ArithToEmitCPass().apply(ctx, module)
    print("  Applying MemrefLoadToEmitcPass...", flush=True)
    MemrefLoadToEmitcPass().apply(ctx, module)
    print("  Applying MemrefStoreToEmitcPass...", flush=True)
    MemrefStoreToEmitcPass().apply(ctx, module)
    print("  Applying SCFToEmitCPass...", flush=True)
    SCFToEmitCPass().apply(ctx, module)
    print("  Applying RVVToEmitCPass...", flush=True)
    RVVToEmitCPass().apply(ctx, module)
    print("  Passes done.", flush=True)

    emitc_mlir = print_module(module)

    # 3. Show EmitC MLIR + verify intrinsic names
    print("── [2] EmitC MLIR after lowering ─────────────────────────")
    ok = verify_emitc_output(emitc_mlir)

    # 4. Translate EmitC → C via mlir-translate
    print("── [3] Generated C (via mlir-translate -mlir-to-cpp) ─────")
    c_code = translate_to_c(module)
    if c_code:
        print(c_code)
        print(f"── Written to: {C_TMP}")
    else:
        print("  ⚠️  C generation skipped (mlir-translate unavailable or failed)")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
