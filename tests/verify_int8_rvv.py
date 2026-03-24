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
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    FunctionType,
)
from xdsl.ir import Block, Region
from xdsl.printer import Printer

# ── Dialect + Pass imports ───────────────────────────────────────────────────
from xdsltemplate.dialects.rvv import (
    RVV,
    RVVInt8M1Type,
    RVVInt16M2Type,
    RVVInt32M4Type,
    vle8_v_i8m1Op,
    vwadd_vx_i16m2Op,
    vmv_v_x_i32m4Op,
    vle32_v_i32m4Op,
    vwmacc_vx_i32m4Op,
    vse32_v_i32m4Op,
)
from xdsltemplate.transforms.rvv_to_emitc import RVVToEmitCPass

# ── Reference intrinsic names (from the hand-written C) ──────────────────────
EXPECTED_INTRINSICS = [
    "__riscv_vmv_v_x_i32m4",    # zero-init accumulator
    "__riscv_vle8_v_i8m1",      # load A
    "__riscv_vwadd_vx_i16m2",   # widen A: i8 → i16
    "__riscv_vwmacc_vx_i32m4",  # widening MAC
    "__riscv_vle32_v_i32m4",    # load C (beta≠0)
    "__riscv_vse32_v_i32m4",    # store C
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
    """
    Builds a minimal func.func containing ONE of each new INT8 RVV op.
    Corresponds to one iteration of the K-loop body in the reference kernel:

        vint32m4_t C00  = __riscv_vmv_v_x_i32m4(0, vlength);        // zero-init
        vint32m4_t C00b = __riscv_vle32_v_i32m4(&Cr[0], vlength);   // beta≠0 load
        vint8m1_t  vtmp = __riscv_vle8_v_i8m1(&Ar[0], vlength);     // load A
        vint16m2_t A0   = __riscv_vwadd_vx_i16m2(vtmp, 0, vlength); // widen
        C00 = __riscv_vwmacc_vx_i32m4(C00, Br[0], A0, vlength);     // MAC
        __riscv_vse32_v_i32m4(&Cr[0], C00, vlength);                 // store
    """
    index_t    = IndexType()
    i8_t       = IntegerType(8)
    i32_t      = IntegerType(32)
    memref_i8  = MemRefType(i8_t,  [-1])
    memref_i32 = MemRefType(i32_t, [-1])

    # Function: (Ar: memref<?xi8>, Br: memref<?xi8>, Cr: memref<?xi32>) -> ()
    func_type   = FunctionType.from_lists([memref_i8, memref_i8, memref_i32], [])
    entry_block = Block(arg_types=[memref_i8, memref_i8, memref_i32])
    Ar, Br, Cr  = entry_block.args

    # Constants
    c_vl  = arith.ConstantOp(IntegerAttr(32, index_t));  entry_block.add_op(c_vl)
    c0i   = arith.ConstantOp(IntegerAttr(0,  index_t));  entry_block.add_op(c0i)
    c0i8  = arith.ConstantOp(IntegerAttr(0,  i8_t));     entry_block.add_op(c0i8)
    c0i32 = arith.ConstantOp(IntegerAttr(0,  i32_t));    entry_block.add_op(c0i32)

    vl  = c_vl.result
    i0  = c0i.result
    i80 = c0i8.result
    i320= c0i32.result

    # 1. vmv_v_x_i32m4: zero-init  →  C00 = __riscv_vmv_v_x_i32m4(0, vl)
    vmv_op = vmv_v_x_i32m4Op(i320, vl);          entry_block.add_op(vmv_op)

    # 2. vle32_v_i32m4: load C (beta≠0)  →  C00b = __riscv_vle32_v_i32m4(&Cr[0], vl)
    vle32i32_op = vle32_v_i32m4Op(Cr, i0, vl);   entry_block.add_op(vle32i32_op)

    # 3. vle8_v_i8m1: load A  →  vtmp = __riscv_vle8_v_i8m1(&Ar[0], vl)
    vle8_op = vle8_v_i8m1Op(Ar, i0, vl);         entry_block.add_op(vle8_op)

    # 4. vwadd_vx_i16m2: widen  →  A0 = __riscv_vwadd_vx_i16m2(vtmp, 0, vl)
    vwadd_op = vwadd_vx_i16m2Op(vle8_op.result, i80, vl)
    entry_block.add_op(vwadd_op)

    # 5. vwmacc_vx_i32m4: MAC  →  C00 = __riscv_vwmacc_vx_i32m4(C00, Br[0], A0, vl)
    br0 = memref.LoadOp.get(Br, [i0]);            entry_block.add_op(br0)
    vwmacc_op = vwmacc_vx_i32m4Op(
        vmv_op.result, br0.res, vwadd_op.result, vl
    )
    entry_block.add_op(vwmacc_op)

    # 6. vse32_v_i32m4: store  →  __riscv_vse32_v_i32m4(&Cr[0], C00, vl)
    store_op = vse32_v_i32m4Op(Cr, i0, vwmacc_op.result, vl)
    entry_block.add_op(store_op)

    entry_block.add_op(func.ReturnOp())

    gemm_func = func.FuncOp("test_int8_ops", func_type, Region([entry_block]))
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
OUT_DIR        = "/home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/RVV_code_gen_via_MLIR_xDSL/tests/tmp_int_codeGen"
MLIR_TMP       = f"{OUT_DIR}/verify_int8_rvv.mlir"
C_TMP          = f"{OUT_DIR}/verify_int8_rvv.c"

# ─────────────────────────────────────────────────────────────────────────────
def translate_to_c(module: ModuleOp) -> str | None:
    """Write Generic EmitC MLIR to OUT_DIR and call mlir-translate -mlir-to-cpp."""
    import subprocess, os
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Needs to be generic format for mlir-translate to parse
    buf = StringIO()
    Printer(stream=buf, print_generic_format=True).print_op(module)
    emitc_mlir_generic = buf.getvalue().replace("<-1x", "<?x")
    
    with open(MLIR_TMP, "w") as f:
        f.write(emitc_mlir_generic)
    try:
        result = subprocess.run(
            [MLIR_TRANSLATE, "-allow-unregistered-dialect",
             "-mlir-to-cpp", MLIR_TMP, "-o", C_TMP],
            capture_output=True, text=True, timeout=30,
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
from xdsltemplate.transforms.memref_to_emitc import MemRefToEmitCPass
from xdsltemplate.transforms.arith_to_emitc import ArithToEmitCPass
from xdsltemplate.transforms.memref_load_to_emitc import MemrefLoadToEmitcPass
from xdsltemplate.transforms.memref_store_to_emitc import MemrefStoreToEmitcPass

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
