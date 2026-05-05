"""
pack_builder.py

Shared, dtype-parameterised IR builder for pack_CB_v and pack_RB_v.

Matches the BLIS reference design in gemm_blis.c exactly:
  pack_CB( orderM, transM, mc, nc, M, ldM, Mc, RR )
  pack_RB( orderM, transM, mc, nc, M, ldM, Mc, RR )

Each builder function takes:
  - RR         : int  — block size (NR for CB, MR for RB); runtime-passable
  - elem_bytes : int  — element size in bytes (4=fp32, 2=fp16)
  - suffix     : str  — name suffix for the generated function (e.g. "fp32")
  - load_type  : xDSL type for the vector register (e.g. RVVFloat32M1Type())
  - memref_type: xDSL type for the flat memref

Generated C function names encode block size and dtype:
  pack_CB_v_if_{suffix}_RR{RR}    — if-branch (strided load)
  pack_CB_v_else_{suffix}_RR{RR}  — else-branch (unit-stride load)
  pack_RB_v_if_{suffix}_RR{RR}    — if-branch (unit-stride load)
  pack_RB_v_else_{suffix}_RR{RR}  — else-branch (strided load)

Memory access derivation from gemm_blis.c Mcol(a1,a2) = M[(a2)*ldM + (a1)]:
  pack_CB if:   Mc[j*mc + i*RR + jj] = M[(j+jj)*ldM + i]   → stride=ldM → vlse32
  pack_CB else: Mc[j*mc + i*RR + jj] = M[i*ldM + (j+jj)]   → stride=1   → vle32
  pack_RB if:   Mc[i*nc + j*RR + ii] = M[j*ldM + (i+ii)]   → stride=1   → vle32
  pack_RB else: Mc[i*nc + j*RR + ii] = M[(i+ii)*ldM + j]   → stride=ldM → vlse32
"""

from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.ir import Block, Region, SSAValue

from xdsltemplate.dialects.rvv import (
    SetvlOp,
    vle32_v_f32m1Op,
    vlse32_v_f32m1Op,
    vse32_v_f32m1Op,
)


# ── Internal helpers ────────────────────────────────────────────────────────────

def _const(val: int, idx: IndexType) -> arith.ConstantOp:
    return arith.ConstantOp(IntegerAttr(val, idx))


def _make_fn(name: str, memref_type, idx: IndexType):
    """Create a func.FuncOp with signature (M*, Mc*, mc, nc, ldM) -> ()."""
    fn = func.FuncOp(name, ([memref_type, memref_type, idx, idx, idx], []))
    return fn, fn.args


# ── pack_CB_v — if-branch: vlse32 (strided, Col-Major Non-Trans) ─────────────
#
# Reference (gemm_blis.c, pack_CB if-branch):
#   for j = 0..nc step RR:
#     for i = 0..mc step 1:
#       vec = { M[(j+0)*ldM+i], M[(j+1)*ldM+i], ..., M[(j+nr-1)*ldM+i] }
#       Mc[j*mc + i*RR .. j*mc + i*RR + nr-1] = vec
#   stride between consecutive vector elements = ldM * elem_bytes
#
def build_pack_CB_v_if(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(f"pack_CB_v_if_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]

    c0  = _const(0, idx);  c1 = _const(1, idx)
    cEB = _const(elem_bytes, idx);  cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cEB, cRR])

    # Outer: j = 0..nc step RR
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cRR.result, [], Region([ob])))
    j = ob.args[0]

    rem    = arith.SubiOp(nc, j)
    nr     = arith.MinSIOp(rem.result, cRR.result)
    vl_op  = SetvlOp(nr.result)
    bstr   = arith.MuliOp(ldM, cEB.result)        # byte stride = ldM * elem_bytes
    ob.add_ops([rem, nr, vl_op, bstr])

    # Inner: i = 0..mc step 1
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, mc, c1.result, [], Region([ib])))
    i = ib.args[0]

    # src_off = j * ldM + i
    j_ldM   = arith.MuliOp(j, ldM)
    src_off = arith.AddiOp(j_ldM.result, i)
    load_op = vlse32_v_f32m1Op(M, src_off.result, bstr.result, vl_op.vl)

    # dst_off = j * mc + i * RR
    j_mc    = arith.MuliOp(j, mc)
    i_RR    = arith.MuliOp(i, cRR.result)
    dst_off = arith.AddiOp(j_mc.result, i_RR.result)
    store   = vse32_v_f32m1Op(Mc, dst_off.result, load_op.result, vl_op.vl)

    ib.add_ops([j_ldM, src_off, load_op, j_mc, i_RR, dst_off, store, scf.YieldOp()])
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


# ── pack_CB_v — else-branch: vle32 (unit-stride, Row-Major Non-Trans) ─────────
#
# Reference (gemm_blis.c, pack_CB else-branch):
#   for j = 0..nc step RR:
#     for i = 0..mc step 1:
#       vec = { M[i*ldM+j], M[i*ldM+j+1], ..., M[i*ldM+j+nr-1] }
#       Mc[j*mc + i*RR .. j*mc + i*RR + nr-1] = vec
#   consecutive elements are contiguous → unit-stride vle32
#
def build_pack_CB_v_else(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(f"pack_CB_v_else_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]

    c0 = _const(0, idx);  c1 = _const(1, idx);  cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cRR])

    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cRR.result, [], Region([ob])))
    j = ob.args[0]

    rem   = arith.SubiOp(nc, j)
    nr    = arith.MinSIOp(rem.result, cRR.result)
    vl_op = SetvlOp(nr.result)
    ob.add_ops([rem, nr, vl_op])

    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, mc, c1.result, [], Region([ib])))
    i = ib.args[0]

    # src_off = i * ldM + j
    i_ldM   = arith.MuliOp(i, ldM)
    src_off = arith.AddiOp(i_ldM.result, j)
    load_op = vle32_v_f32m1Op(M, src_off.result, vl_op.vl)

    # dst_off = j * mc + i * RR
    j_mc    = arith.MuliOp(j, mc)
    i_RR    = arith.MuliOp(i, cRR.result)
    dst_off = arith.AddiOp(j_mc.result, i_RR.result)
    store   = vse32_v_f32m1Op(Mc, dst_off.result, load_op.results[0], vl_op.vl)

    ib.add_ops([i_ldM, src_off, load_op, j_mc, i_RR, dst_off, store, scf.YieldOp()])
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


# ── pack_RB_v — if-branch: vle32 (unit-stride, Col-Major Non-Trans) ──────────
#
# Reference (gemm_blis.c, pack_RB if-branch):
#   for i = 0..mc step RR:
#     for j = 0..nc step 1:
#       vec = { M[j*ldM+i], M[j*ldM+i+1], ..., M[j*ldM+i+rr-1] }
#       Mc[i*nc + j*RR .. i*nc + j*RR + rr-1] = vec
#   for fixed j, elements M[j*ldM+i], M[j*ldM+i+1]... are contiguous → vle32
#
def build_pack_RB_v_if(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(f"pack_RB_v_if_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]

    c0 = _const(0, idx);  c1 = _const(1, idx);  cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cRR])

    # Outer: i = 0..mc step RR  (row blocks)
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cRR.result, [], Region([ob])))
    i = ob.args[0]

    rem   = arith.SubiOp(mc, i)
    rr    = arith.MinSIOp(rem.result, cRR.result)
    vl_op = SetvlOp(rr.result)
    ob.add_ops([rem, rr, vl_op])

    # Inner: j = 0..nc step 1  (all columns)
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, nc, c1.result, [], Region([ib])))
    j = ib.args[0]

    # src_off = j * ldM + i  → M[j*ldM+i..j*ldM+i+rr-1]  unit-stride
    j_ldM   = arith.MuliOp(j, ldM)
    src_off = arith.AddiOp(j_ldM.result, i)
    load_op = vle32_v_f32m1Op(M, src_off.result, vl_op.vl)

    # dst_off = i * nc + j * RR
    i_nc    = arith.MuliOp(i, nc)
    j_RR    = arith.MuliOp(j, cRR.result)
    dst_off = arith.AddiOp(i_nc.result, j_RR.result)
    store   = vse32_v_f32m1Op(Mc, dst_off.result, load_op.results[0], vl_op.vl)

    ib.add_ops([j_ldM, src_off, load_op, i_nc, j_RR, dst_off, store, scf.YieldOp()])
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


# ── pack_RB_v — else-branch: vlse32 (strided, Row-Major Non-Trans) ───────────
#
# Reference (gemm_blis.c, pack_RB else-branch):
#   for i = 0..mc step RR:
#     for j = 0..nc step 1:
#       vec = { M[i*ldM+j], M[(i+1)*ldM+j], ..., M[(i+rr-1)*ldM+j] }
#       Mc[i*nc + j*RR .. i*nc + j*RR + rr-1] = vec
#   for fixed j, elements at rows i..i+rr-1 are separated by ldM → vlse32
#
def build_pack_RB_v_else(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(f"pack_RB_v_else_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]

    c0  = _const(0, idx);  c1 = _const(1, idx)
    cEB = _const(elem_bytes, idx);  cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cEB, cRR])

    # Outer: i = 0..mc step RR
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cRR.result, [], Region([ob])))
    i = ob.args[0]

    rem   = arith.SubiOp(mc, i)
    rr    = arith.MinSIOp(rem.result, cRR.result)
    vl_op = SetvlOp(rr.result)
    bstr  = arith.MuliOp(ldM, cEB.result)          # byte stride = ldM * elem_bytes
    ob.add_ops([rem, rr, vl_op, bstr])

    # Inner: j = 0..nc step 1
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, nc, c1.result, [], Region([ib])))
    j = ib.args[0]

    # src_off = i * ldM + j  → M[(i+0)*ldM+j], stride=ldM
    i_ldM   = arith.MuliOp(i, ldM)
    src_off = arith.AddiOp(i_ldM.result, j)
    load_op = vlse32_v_f32m1Op(M, src_off.result, bstr.result, vl_op.vl)

    # dst_off = i * nc + j * RR
    i_nc    = arith.MuliOp(i, nc)
    j_RR    = arith.MuliOp(j, cRR.result)
    dst_off = arith.AddiOp(i_nc.result, j_RR.result)
    store   = vse32_v_f32m1Op(Mc, dst_off.result, load_op.result, vl_op.vl)

    ib.add_ops([i_ldM, src_off, load_op, i_nc, j_RR, dst_off, store, scf.YieldOp()])
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


# ── Module builders ────────────────────────────────────────────────────────────

def build_pack_CB_module(idx: IndexType, memref_type,
                         RR: int = 8, elem_bytes: int = 4, suffix: str = "fp32") -> ModuleOp:
    """Build MLIR module with both pack_CB_v branches."""
    return ModuleOp([
        build_pack_CB_v_if  (idx, memref_type, RR, elem_bytes, suffix),
        build_pack_CB_v_else(idx, memref_type, RR, elem_bytes, suffix),
    ])


def build_pack_module(idx: IndexType, memref_type,
                         NR: int = 8, MR: int = 4, elem_bytes: int = 4, suffix: str = "fp32") -> ModuleOp:
    """Build MLIR module with ALL packing branches (CB and RB)."""
    return ModuleOp([
        build_pack_CB_v_if  (idx, memref_type, NR, elem_bytes, suffix),
        build_pack_CB_v_else(idx, memref_type, NR, elem_bytes, suffix),
        build_pack_RB_v_if  (idx, memref_type, MR, elem_bytes, suffix),
        build_pack_RB_v_else(idx, memref_type, MR, elem_bytes, suffix),
    ])
