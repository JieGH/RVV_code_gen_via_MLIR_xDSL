"""
8x8 i8*i8->i32 mixed-precision GEMM microkernel.

Equivalent structure to xdsl_api_gemm_RVV_8x8_b0_col_fp32, but for INT8 inputs.

Key design decisions:
  - VLEN=256 bits, mf4 types → exactly 8 elements per vector register
  - vl = setvl_mf4(8) = 8 (fixed constant, not runtime argument)
  - Accumulators: vint32m1_t (safe for BERT Large kc up to ~133,000)
  - Inner loop: 2 instructions per column per k iteration:
      tmp_i16 = vwmacc_vx_i16mf2(zero_i16, b_scalar, A_vec, vl)   # i8*i8 -> i16 product
      acc[j]  = vwadd_wv_i32m1(acc[j], tmp_i16, vl)                # widen i16 -> i32, accumulate

Run:
    PYTHONPATH=src .venv/bin/python tests/test_mixed_precision_macc.py > /tmp/macc_out.txt 2>&1 &
    # wait ~30s then:
    cat /tmp/macc_out.txt
"""

import subprocess
from io import StringIO

from xdsl.context import Context
from xdsl.dialects import arith, func, memref, scf
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

from xdsltemplate.dialects.rvv import (
    RVVInt32M1Type,
    SetvlMf4Op,
    vle8_v_i8mf4Op,
    vmv_v_x_i16mf2Op,
    vmv_v_x_i32m1Op,
    vse32_v_i32m1Op,
    vwadd_wv_i32m1Op,
    vwmacc_vx_i16mf2Op,
)
from xdsltemplate.transforms.arith_to_emitc import ArithToEmitCPass
from xdsltemplate.transforms.memref_load_to_emitc import MemrefLoadToEmitcPass
from xdsltemplate.transforms.memref_store_to_emitc import MemrefStoreToEmitcPass
from xdsltemplate.transforms.memref_to_emitc import MemRefToEmitCPass
from xdsltemplate.transforms.rvv_to_emitc import RVVToEmitCPass
from xdsltemplate.transforms.scf_to_emitc import SCFToEmitCPass

NR = 8  # tile columns (= number of A-row elements at VLEN=256, mf4)
KERNEL_NAME = "xdsl_api_gemm_RVV_8x8_i8i8i32_b0"
MLIR_TRANSLATE = (
    "/home/jlei/Documents/ws_MLIR/llvm_pj_dir_06092025"
    "/llvm-project/build/bin/mlir-translate"
)
OUT_DIR = "tmp_int_codeGen"
OUT_MLIR_HR = f"{OUT_DIR}/mixed_precision_macc.mlir"  # human-readable
OUT_MLIR_GEN = f"{OUT_DIR}/mixed_precision_macc_generic.mlir"  # for mlir-translate
OUT_C = f"{OUT_DIR}/mixed_precision_macc.c"


def build_kernel() -> ModuleOp:
    """
    void KERNEL_NAME(
        int kc,
        int8_t* A, int lda,
        int8_t* B, int ldb,
        int32_t* C, int ldc
    )
    """
    idx = IndexType()
    i8 = IntegerType(8)
    i16 = IntegerType(16)
    i32 = IntegerType(32)
    acc_t = RVVInt32M1Type()

    input_types = [
        idx,  # kc
        MemRefType(i8, [-1]),  # A
        idx,  # lda
        MemRefType(i8, [-1]),  # B
        idx,  # ldb
        MemRefType(i32, [-1]),  # C
        idx,  # ldc
    ]
    func_type = FunctionType.from_lists(input_types, [])
    entry = Block(arg_types=input_types)
    kc, Ar, lda, Br, ldb, Cr, ldc = entry.args

    # --- Constants ---
    c0 = arith.ConstantOp(IntegerAttr(0, idx))
    entry.add_op(c0)
    c1 = arith.ConstantOp(IntegerAttr(1, idx))
    entry.add_op(c1)
    c8 = arith.ConstantOp(IntegerAttr(8, idx))
    entry.add_op(c8)  # tile size
    c0i8 = arith.ConstantOp(IntegerAttr(0, i8))
    entry.add_op(c0i8)
    c0i16 = arith.ConstantOp(IntegerAttr(0, i16))
    entry.add_op(c0i16)
    c0i32 = arith.ConstantOp(IntegerAttr(0, i32))
    entry.add_op(c0i32)

    # --- VL: fixed 8 elements (VLEN=256, mf4) ---
    sv = SetvlMf4Op(c8.result)
    entry.add_op(sv)
    vl = sv.vl

    # --- Pre-compute column offset constants (for B indexing and C store) ---
    col_idx = []
    for j in range(NR):
        cj = arith.ConstantOp(IntegerAttr(j, idx))
        entry.add_op(cj)
        col_idx.append(cj.result)

    # --- Zero-init 8 vint32m1_t accumulators ---
    init_accs = []
    for _ in range(NR):
        vmv = vmv_v_x_i32m1Op(c0i32.result, vl)
        entry.add_op(vmv)
        init_accs.append(vmv.result)

    # --- zero_i16 vector reused each inner iteration as neutral element ---
    zero_i16_vec = vmv_v_x_i16mf2Op(c0i16.result, vl)
    entry.add_op(zero_i16_vec)

    # =====================================================================
    # SCF for-loop over k: for k in range(0, kc, 1)
    #   iter_args: 8 vint32m1_t accumulators + bA_offset + bB_offset
    # =====================================================================
    loop_body_types = [idx] + [acc_t] * NR + [idx, idx]
    loop_body = Block(arg_types=loop_body_types)

    # Unpack loop body args
    k_iter = loop_body.args[0]
    l_accs = [loop_body.args[1 + j] for j in range(NR)]
    bA = loop_body.args[NR + 1]  # byte offset into A
    bB = loop_body.args[NR + 2]  # byte offset into B

    # Load A row as vint8mf4_t: A[bA .. bA+7]
    vle = vle8_v_i8mf4Op(Ar, bA, vl)
    loop_body.add_op(vle)
    A_vec = vle.result

    # For each of the NR B columns: load scalar, compute product, accumulate
    new_accs = []
    for j in range(NR):
        # B scalar offset = bB + j
        bj_add = arith.AddiOp(bB, col_idx[j])
        loop_body.add_op(bj_add)
        b_scalar = memref.LoadOp.get(Br, [bj_add.result])
        loop_body.add_op(b_scalar)

        # tmp_i16 = vwmacc_vx_i16mf2(zero_i16, b_scalar, A_vec, vl)
        tmp = vwmacc_vx_i16mf2Op(zero_i16_vec.result, b_scalar.res, A_vec, vl)
        loop_body.add_op(tmp)

        # acc[j] = vwadd_wv_i32m1(acc[j], tmp_i16, vl)
        new_acc = vwadd_wv_i32m1Op(l_accs[j], tmp.result, vl)
        loop_body.add_op(new_acc)
        new_accs.append(new_acc.result)

    # Advance offsets: bA += lda, bB += ldb
    next_bA = arith.AddiOp(bA, lda)
    loop_body.add_op(next_bA)
    next_bB = arith.AddiOp(bB, ldb)
    loop_body.add_op(next_bB)

    # Yield updated accumulators + offsets
    loop_body.add_op(scf.YieldOp(*new_accs, next_bA.result, next_bB.result))

    # ForOp: iter_args = acc0..7, bA=0, bB=0
    iter_inits = init_accs + [c0.result, c0.result]
    for_op = scf.ForOp(c0.result, kc, c1.result, iter_inits, Region([loop_body]))
    entry.add_op(for_op)

    # =====================================================================
    # STORE: for j in 0..7: C[j * ldc] = acc[j]
    # =====================================================================
    final_accs = [for_op.results[j] for j in range(NR)]
    for j in range(NR):
        col_offset = arith.MuliOp(col_idx[j], ldc)
        entry.add_op(col_offset)
        vse = vse32_v_i32m1Op(Cr, col_offset.result, final_accs[j], vl)
        entry.add_op(vse)

    entry.add_op(func.ReturnOp())

    gemm_func = func.FuncOp(KERNEL_NAME, func_type, Region([entry]))
    return ModuleOp([gemm_func])


def lower_to_emitc(module: ModuleOp, ctx: Context) -> ModuleOp:
    """Apply the same pass pipeline as test_codeGeneration.py."""
    for p in [
        MemRefToEmitCPass(),
        ArithToEmitCPass(),
        MemrefLoadToEmitcPass(),
        MemrefStoreToEmitcPass(),
        SCFToEmitCPass(),
        RVVToEmitCPass(),
    ]:
        p.apply(ctx, module)
    return module


if __name__ == "__main__":
    # --- Build ---
    module = build_kernel()
    print("=== [1] MLIR (before lowering) ===")
    Printer().print_op(module)
    print("\n")

    # --- Lower ---
    ctx = Context()
    module = lower_to_emitc(module, ctx)
    print("=== [2] EmitC (after lowering) ===")
    Printer().print_op(module)
    print("\n")

    # --- Verify expected intrinsics ---
    buf = StringIO()
    Printer(stream=buf).print_op(module)
    emitc_str = buf.getvalue()
    EXPECTED = [
        "__riscv_vsetvl_e8mf4",
        "__riscv_vle8_v_i8mf4",
        "__riscv_vmv_v_x_i16mf2",
        "__riscv_vwmacc_vx_i16mf2",
        "__riscv_vwadd_wv_i32m1",
        "__riscv_vmv_v_x_i32m1",
        "__riscv_vse32_v_i32m1",
    ]
    print("=== [3] Intrinsic check ===")
    all_ok = True
    for fn in EXPECTED:
        ok = fn in emitc_str
        print(f"  {'OK  ' if ok else 'MISS'}: {fn}")
        if not ok:
            all_ok = False
    print()

    # --- Write human-readable MLIR (EmitC IR) ---
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_MLIR_HR, "w") as f:
        Printer(stream=f).print_op(module)
    print(f"Wrote human-readable MLIR → {OUT_MLIR_HR}")

    # --- Write generic MLIR for mlir-translate ---
    with open(OUT_MLIR_GEN, "w") as f:
        Printer(stream=f, print_generic_format=True).print_op(module)
    print(f"Wrote generic MLIR       → {OUT_MLIR_GEN}")

    # --- Translate to C via mlir-translate ---
    print(f"\n=== [4] mlir-translate → {OUT_C} ===")
    try:
        r = subprocess.run(
            [
                MLIR_TRANSLATE,
                "-allow-unregistered-dialect",
                "-mlir-to-cpp",
                OUT_MLIR_GEN,
                "-o",
                OUT_C,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            print("mlir-translate FAILED:")
            print(r.stderr)
        else:
            print(f"Wrote C output           → {OUT_C}")
            with open(OUT_C) as f:
                print(f.read())
    except FileNotFoundError:
        print(f"mlir-translate not found at: {MLIR_TRANSLATE}")

    raise SystemExit(0 if all_ok else 1)
