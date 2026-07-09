"""
generate_packing.py

Unified entry point for generating all vectorized packing kernels (CB and RB).
Produces a single C file containing all four branches.

Generated C functions (default NR=8, MR=8):
  pack_CB_v_if_fp32_RR8
  pack_CB_v_else_fp32_RR8
  pack_RB_v_if_fp32_RR8
  pack_RB_v_else_fp32_RR8
"""

import argparse
import os
import subprocess
import sys
from io import StringIO

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, script_dir)

import pack_builder as pb
from xdsl.context import Context
from xdsl.dialects import arith, builtin, emitc, func, memref, scf
from xdsl.dialects.builtin import Float32Type, IndexType, MemRefType, IntegerType, f16
from xdsl.printer import Printer

from xdsltemplate.dialects.rvv import RVV
from xdsltemplate.transforms.arith_to_emitc import ArithToEmitCPass
from xdsltemplate.transforms.memref_load_to_emitc import MemrefLoadToEmitcPass
from xdsltemplate.transforms.memref_store_to_emitc import MemrefStoreToEmitcPass
from xdsltemplate.transforms.memref_to_emitc import MemRefToEmitCPass
from xdsltemplate.transforms.rvv_to_emitc import RVVToEmitCPass
from xdsltemplate.transforms.scf_to_emitc import SCFToEmitCPass


def make_context() -> Context:
    ctx = Context()
    for d in [
        builtin.Builtin,
        func.Func,
        memref.MemRef,
        arith.Arith,
        scf.Scf,
        emitc.EmitC,
        RVV,
    ]:
        ctx.load_dialect(d)
    return ctx


def print_ir(module) -> str:
    buf = StringIO()
    Printer(stream=buf).print(module)
    return buf.getvalue()


def verify_intrinsics(
    output: str, fn_name: str, expected: list, forbidden: list
) -> bool:
    ok = True
    print(f"  [{fn_name}]")
    if not output:
        print("    ❌  FUNCTION NOT FOUND IN IR")
        return False
    for intr in expected:
        if intr in output:
            print(f"    ✅  {intr}")
        else:
            print(f"    ❌  {intr}  ← MISSING")
            ok = False
    for intr in forbidden:
        if intr not in output:
            print(f"    ✅  {intr} — correctly absent")
        else:
            print(f"    ❌  {intr}  ← must be absent")
            ok = False
    return ok


def run(
    NR: int,
    MR: int,
    elem_bytes: int,
    suffix: str,
    output: str,
    mlir_translate: str | None,
    vlen_bits: int = 256,
    method: str = "AXPY_FP32_Pack",
):
    idx = IndexType()
    if suffix == "i8":
        elem_type = IntegerType(8)
    elif suffix == "i32":
        elem_type = IntegerType(32)
    elif suffix == "fp16":
        elem_type = f16
    else:
        elem_type = Float32Type()
    
    memref_ty = MemRefType(elem_type, [-1])
    module = pb.build_pack_module(
        idx, memref_ty, NR=NR, MR=MR, elem_bytes=elem_bytes, suffix=suffix, vlen_bits=vlen_bits, method=method
    )
    ctx = make_context()

    print("=" * 60)
    print(f"  Unified Packing Generation ({suffix}, NR={NR}, MR={MR})")
    print("=" * 60)

    MemRefToEmitCPass().apply(ctx, module)
    ArithToEmitCPass().apply(ctx, module)
    MemrefLoadToEmitcPass().apply(ctx, module)
    MemrefStoreToEmitcPass().apply(ctx, module)
    SCFToEmitCPass().apply(ctx, module)
    RVVToEmitCPass().apply(ctx, module)

    ir_text = print_ir(module)

    # Split IR by function to verify each independently
    parts = ir_text.split("func.func @")
    fn_blocks = {}
    for part in parts[1:]:
        name = part.split("(")[0].strip()
        fn_blocks[name] = part

    print("\n  Intrinsic Verification")
    print("-" * 30)
    all_ok = True

    cb_if = f"pack_CB_v_if_{method}_{suffix}_RR{NR}"
    cb_else = f"pack_CB_v_else_{method}_{suffix}_RR{NR}"
    rb_if = f"pack_RB_v_if_{method}_{suffix}_RR{MR}"
    rb_else = f"pack_RB_v_else_{method}_{suffix}_RR{MR}"

    if method == "DOT_INT8_LANE_Pack":
        # B-side (CB) uses 8-bit strided loads and 8-bit strided stores
        all_ok &= verify_intrinsics(fn_blocks.get(cb_if), cb_if, ["__riscv_vlse8", "__riscv_vsse8"], [])
        # A-side (RB) uses 8-bit loads
        all_ok &= verify_intrinsics(fn_blocks.get(rb_if), rb_if, ["__riscv_vlse8"], [])
        # _else branches are intentionally empty for this method, skip checking them
    else:
        all_ok &= verify_intrinsics(
            fn_blocks.get(cb_if), cb_if, ["__riscv_vlse32"], ["__riscv_vle32"]
        )
        all_ok &= verify_intrinsics(
            fn_blocks.get(cb_else), cb_else, ["__riscv_vle32"], ["__riscv_vlse32"]
        )
        all_ok &= verify_intrinsics(
            fn_blocks.get(rb_if), rb_if, ["__riscv_vle32"], ["__riscv_vlse32"]
        )
        all_ok &= verify_intrinsics(
            fn_blocks.get(rb_else), rb_else, ["__riscv_vlse32"], ["__riscv_vle32"]
        )

    print("\n  ✅  PASS" if all_ok else "\n  ❌  FAIL")
    if not all_ok:
        sys.exit(1)

    if not mlir_translate:
        print("\nSkipping C translation.")
        return

    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp_mlir = output + ".mlir"
    with open(tmp_mlir, "w") as f:
        Printer(stream=f, print_generic_format=True).print(module)

    cmd = [mlir_translate, "--mlir-to-cpp", tmp_mlir, "-o", output]
    print(f"\nRunning: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(output) as f:
            content = f.read()
        with open(output, "w") as f:
            f.write('#include <riscv_vector.h>\n#include <algorithm>\n\nextern "C" {\n')
            f.write(content)
            f.write("\n}\n")
        print(f"✅ Generated: {output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ mlir-translate failed:\n{e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mlir-translate", type=str, default=None)
    p.add_argument("--output", type=str, default="tests/packing/gen/pack_v_fp32.c")
    p.add_argument("--NR", type=int, default=8)
    p.add_argument("--MR", type=int, default=8)
    p.add_argument("--elem-bytes", type=int, default=4)
    p.add_argument("--suffix", type=str, default="fp32")
    p.add_argument("--vlen", type=int, default=256, help="Hardware VLEN in bits (default 256 for SpacemiT K1)")
    p.add_argument("--method", type=str, default="AXPY_FP32_Pack", help="Packing method identifier")
    args = p.parse_args()
    run(
        args.NR, args.MR, args.elem_bytes, args.suffix, args.output, args.mlir_translate, vlen_bits=args.vlen, method=args.method
    )
