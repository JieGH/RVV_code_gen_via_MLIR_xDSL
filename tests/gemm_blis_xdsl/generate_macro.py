"""
generate_macro.py

Entry point for generating the GEMM Macro-Kernel with layout specialization.
"""

import argparse
import os
import sys
import subprocess
from io import StringIO

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, script_dir)

from xdsl.context import Context
from xdsl.dialects import arith, builtin, emitc, func, memref, scf
from xdsl.dialects.builtin import ModuleOp, IndexType, MemRefType, Float32Type
from xdsl.printer import Printer

from xdsltemplate.dialects.rvv import RVV
from xdsltemplate.transforms.arith_to_emitc        import ArithToEmitCPass
from xdsltemplate.transforms.memref_load_to_emitc  import MemrefLoadToEmitcPass
from xdsltemplate.transforms.memref_store_to_emitc import MemrefStoreToEmitcPass
from xdsltemplate.transforms.memref_to_emitc       import MemRefToEmitCPass
from xdsltemplate.transforms.rvv_to_emitc          import RVVToEmitCPass
from xdsltemplate.transforms.scf_to_emitc          import SCFToEmitCPass

import macro_builder as mb
import pack_builder as pb
import blis_model

def make_context() -> Context:
    ctx = Context()
    for d in [builtin.Builtin, func.Func, memref.MemRef,
              arith.Arith, scf.Scf, emitc.EmitC, RVV]:
        ctx.load_dialect(d)
    return ctx

def run(MC, NC, KC, MR, NR, orderA, transA, orderB, transB, orderC, suffix, output, mlir_translate):
    module = mb.build_macro_module(
        MC=MC, NC=NC, KC=KC, MR=MR, NR=NR,
        orderA=orderA, transA=transA,
        orderB=orderB, transB=transB,
        orderC=orderC,
        suffix=suffix
    )
    
    idx_type = IndexType()
    memref_type = MemRefType(Float32Type(), [-1])
    pack_module = pb.build_pack_module(idx_type, memref_type, NR=NR, MR=MR, elem_bytes=4, suffix=suffix)
    
    # Merge packing functions into the main module BEFORE the macro function
    pack_ops = list(pack_module.body.blocks[0].ops)
    for op in pack_ops:
        op.detach()
    # Insert at the beginning of the block so they are defined before being called
    first_op = next(iter(module.body.blocks[0].ops))
    module.body.blocks[0].insert_ops_before(pack_ops, first_op)

    # Generate cache configuration header for the C++ driver
    config_header = os.path.join(os.path.dirname(output), "cache_config.h")
    os.makedirs(os.path.dirname(config_header), exist_ok=True)
    with open(config_header, "w") as f:
        f.write("// Auto-generated cache configuration\n")
        f.write(f"#define MC_OPT {MC}\n")
        f.write(f"#define NC_OPT {NC}\n")
        f.write(f"#define KC_OPT {KC}\n")
        f.write(f"#define MR_OPT {MR}\n")
        f.write(f"#define NR_OPT {NR}\n")
    print(f"✅ Generated config header: {config_header}")

    ctx = make_context()

    print("=" * 60)
    print(f"  GEMM Macro-Kernel & Packing Generation")
    print(f"  A: {orderA}{transA}, B: {orderB}{transB}, C: {orderC}")
    print("=" * 60)

    MemRefToEmitCPass().apply(ctx, module)
    ArithToEmitCPass().apply(ctx, module)
    MemrefLoadToEmitcPass().apply(ctx, module)
    MemrefStoreToEmitcPass().apply(ctx, module)
    SCFToEmitCPass().apply(ctx, module)
    RVVToEmitCPass().apply(ctx, module)

    if not mlir_translate:
        buf = StringIO()
        Printer(stream=buf).print(module)
        print(buf.getvalue())
        return

    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp_mlir = output + ".mlir"
    with open(tmp_mlir, "w") as f:
        Printer(stream=f, print_generic_format=True).print(module)

    cmd = [mlir_translate, "--mlir-to-cpp", tmp_mlir, "-o", output]
    print(f"\nRunning: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(output) as f: content = f.read()
        with open(output, "w") as f:
            f.write("#include <riscv_vector.h>\n#include <algorithm>\n#include <stdint.h>\n\n")
            f.write("#define GET_PTR(ptr, off) (ptr + off)\n")
            f.write("#define GET_BETA(pc, beta) (pc == 0 ? beta : 1.0f)\n\n")
            f.write("extern \"C\" {\n")
            f.write("void gemm_microkernel_v(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac_ptr, float* Bc_ptr, float betaI, float* Cptr, int32_t ldC);\n\n")
            f.write(content)
            f.write("\n}\n")
        print(f"✅ Generated: {output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ mlir-translate failed:\n{e.stderr}")
        sys.exit(1)

def parse_model_dims(model_path):
    """Parse a .dat model file to find the maximum M, N, K dimensions."""
    m_max, n_max, k_max = 0, 0, 0
    try:
        with open(model_path, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    # Format: Layer M N K ...
                    m, n, k = int(parts[1]), int(parts[2]), int(parts[3])
                    m_max = max(m_max, m)
                    n_max = max(n_max, n)
                    k_max = max(k_max, k)
    except Exception as e:
        print(f"[WARNING] Could not parse model file {model_path}: {e}")
    return m_max, n_max, k_max

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mlir-translate", type=str, default=None)
    p.add_argument("--output", type=str, default="tests/gemm_blis/gen/gemm_macro_v.c")
    p.add_argument("--MC", type=int, default=256)
    p.add_argument("--NC", type=int, default=1024)
    p.add_argument("--KC", type=int, default=256)
    p.add_argument("--MR", type=int, default=4)
    p.add_argument("--NR", type=int, default=8)
    p.add_argument("--orderA", type=str, default='C')
    p.add_argument("--transA", type=str, default='N')
    p.add_argument("--orderB", type=str, default='C')
    p.add_argument("--transB", type=str, default='N')
    p.add_argument("--orderC", type=str, default='C')
    p.add_argument("--suffix", type=str, default="fp32")
    p.add_argument("--arch", type=str, default=None, help="Target architecture for analytical modeling (e.g., k1)")
    p.add_argument("--m_max", type=int, default=512)
    p.add_argument("--n_max", type=int, default=512)
    p.add_argument("--k_max", type=int, default=512)
    p.add_argument("--model", type=str, default=None, help="Path to CNN model file to derive max dimensions")
    args = p.parse_args()
    
    # 1. Detect MR/NR from kernels directory if defaults are used
    if args.MR == 4 and args.NR == 8:
        import os
        import re
        kernel_dir = os.path.join(os.path.dirname(__file__), "kernels")
        if os.path.exists(kernel_dir):
            for f in os.listdir(kernel_dir):
                match = re.search(r"kernels_RVV_(\d+)x(\d+)_fp32\.c", f)
                if match:
                    args.MR = int(match.group(1))
                    args.NR = int(match.group(2))
                    print(f"[INFO] Auto-detected Micro-tile size from kernels: {args.MR}x{args.NR}")
                    break

    m_max, n_max, k_max = args.m_max, args.n_max, args.k_max
    if args.model:
        mm, nm, km = parse_model_dims(args.model)
        if mm > 0:
            m_max, n_max, k_max = mm, nm, km
            print(f"[INFO] Derived max dimensions from model: M={m_max}, N={n_max}, K={k_max}")

    mc, nc, kc = args.MC, args.NC, args.KC
    if args.arch:
        mc, nc, kc = blis_model.get_optim_mc_nc_kc(
            args.arch, 4, m_max, n_max, k_max, args.MR, args.NR
        )
        print(f"[INFO] Analytical Model ({args.arch}) selected: MC={mc}, NC={nc}, KC={kc}")

    run(mc, nc, kc, args.MR, args.NR, 
        args.orderA, args.transA, args.orderB, args.transB, args.orderC,
        args.suffix, args.output, args.mlir_translate)
