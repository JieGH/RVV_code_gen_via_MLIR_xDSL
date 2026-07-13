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
  - vlen       : int  — target machine vector length (e.g. 8 for fp32 on VLEN=256)
"""

from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.ir import Block, Region

from xdsltemplate.dialects.rvv import (
    SetvlOp,
    vle32_v_f32m1Op,
    vlse32_v_f32m1Op,
    vse32_v_f32m1Op,
)

def _const(val: int, idx: IndexType) -> arith.ConstantOp:
    return arith.ConstantOp(IntegerAttr(val, idx))

def _make_fn(name: str, memref_type, idx: IndexType, dim1_name: str = "dim1", dim2_name: str = "dim2"):
    fn = func.FuncOp(name, ([memref_type, memref_type, idx, idx, idx], []))
    return fn, fn.args

def get_safe_chunk_avl(nr_val, v, vlen, idx, ib_block):
    c_chunk = _const(v * vlen, idx)
    c_chunk_end = _const((v + 1) * vlen, idx)
    end = arith.MinSIOp(nr_val, c_chunk_end.result)
    end_safe = arith.MaxSIOp(end.result, c_chunk.result)
    chunk_avl = arith.SubiOp(end_safe.result, c_chunk.result)
    vl_op = SetvlOp(chunk_avl.result)
    ib_block.add_ops([c_chunk, c_chunk_end, end, end_safe, chunk_avl, vl_op])
    return c_chunk, vl_op

def build_pack_CB_v_if(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8):
    fn, (M, Mc, kc, nc, ldM) = _make_fn(f"pack_CB_v_if_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]
    c0, c1, cEB, cRR = _const(0, idx), _const(1, idx), _const(elem_bytes, idx), _const(RR, idx)
    entry.add_ops([c0, c1, cEB, cRR])

    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cRR.result, [], Region([ob])))
    j = ob.args[0]
    rem = arith.SubiOp(nc, j)
    nr = arith.MinSIOp(rem.result, cRR.result)
    bstr = arith.MuliOp(ldM, cEB.result)
    ob.add_ops([rem, nr, bstr])

    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, kc, c1.result, [], Region([ib])))
    i = ib.args[0]

    j_ldM = arith.MuliOp(j, ldM)
    src_off_base = arith.AddiOp(j_ldM.result, i)
    j_kc, i_RR = arith.MuliOp(j, kc), arith.MuliOp(i, cRR.result)
    dst_off_base = arith.AddiOp(j_kc.result, i_RR.result)
    ib.add_ops([j_ldM, src_off_base, j_kc, i_RR, dst_off_base])

    num_vecs = (RR + vlen - 1) // vlen
    for v in range(num_vecs):
        c_chunk, vl_op = get_safe_chunk_avl(nr.result, v, vlen, idx, ib)
        chunk_ldM = arith.MuliOp(c_chunk.result, ldM)
        src_off_chunk = arith.AddiOp(src_off_base.result, chunk_ldM.result)
        load_op = vlse32_v_f32m1Op(M, src_off_chunk.result, bstr.result, vl_op.vl)
        dst_off_chunk = arith.AddiOp(dst_off_base.result, c_chunk.result)
        store = vse32_v_f32m1Op(Mc, dst_off_chunk.result, load_op.result, vl_op.vl)
        ib.add_ops([chunk_ldM, src_off_chunk, load_op, dst_off_chunk, store])

    ib.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn

def build_pack_CB_v_else(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8):
    fn, (M, Mc, kc, nc, ldM) = _make_fn(f"pack_CB_v_else_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]
    c0, c1, cRR = _const(0, idx), _const(1, idx), _const(RR, idx)
    entry.add_ops([c0, c1, cRR])

    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cRR.result, [], Region([ob])))
    j = ob.args[0]
    rem = arith.SubiOp(nc, j)
    nr = arith.MinSIOp(rem.result, cRR.result)
    ob.add_ops([rem, nr])

    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, kc, c1.result, [], Region([ib])))
    i = ib.args[0]

    i_ldM = arith.MuliOp(i, ldM)
    src_off_base = arith.AddiOp(i_ldM.result, j)
    j_kc, i_RR = arith.MuliOp(j, kc), arith.MuliOp(i, cRR.result)
    dst_off_base = arith.AddiOp(j_kc.result, i_RR.result)
    ib.add_ops([i_ldM, src_off_base, j_kc, i_RR, dst_off_base])

    num_vecs = (RR + vlen - 1) // vlen
    for v in range(num_vecs):
        c_chunk, vl_op = get_safe_chunk_avl(nr.result, v, vlen, idx, ib)
        src_off_chunk = arith.AddiOp(src_off_base.result, c_chunk.result)
        load_op = vle32_v_f32m1Op(M, src_off_chunk.result, vl_op.vl)
        dst_off_chunk = arith.AddiOp(dst_off_base.result, c_chunk.result)
        store = vse32_v_f32m1Op(Mc, dst_off_chunk.result, load_op.results[0], vl_op.vl)
        ib.add_ops([src_off_chunk, load_op, dst_off_chunk, store])

    ib.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn

def build_pack_RB_v_if(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8):
    fn, (M, Mc, mc, kc, ldM) = _make_fn(f"pack_RB_v_if_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]
    c0, c1, cRR = _const(0, idx), _const(1, idx), _const(RR, idx)
    entry.add_ops([c0, c1, cRR])

    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cRR.result, [], Region([ob])))
    i = ob.args[0]
    rem = arith.SubiOp(mc, i)
    rr = arith.MinSIOp(rem.result, cRR.result)
    ob.add_ops([rem, rr])

    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, kc, c1.result, [], Region([ib])))
    j = ib.args[0]

    j_ldM = arith.MuliOp(j, ldM)
    src_off_base = arith.AddiOp(j_ldM.result, i)
    i_kc, j_RR = arith.MuliOp(i, kc), arith.MuliOp(j, cRR.result)
    dst_off_base = arith.AddiOp(i_kc.result, j_RR.result)
    ib.add_ops([j_ldM, src_off_base, i_kc, j_RR, dst_off_base])

    num_vecs = (RR + vlen - 1) // vlen
    for v in range(num_vecs):
        c_chunk, vl_op = get_safe_chunk_avl(rr.result, v, vlen, idx, ib)
        src_off_chunk = arith.AddiOp(src_off_base.result, c_chunk.result)
        load_op = vle32_v_f32m1Op(M, src_off_chunk.result, vl_op.vl)
        dst_off_chunk = arith.AddiOp(dst_off_base.result, c_chunk.result)
        store = vse32_v_f32m1Op(Mc, dst_off_chunk.result, load_op.results[0], vl_op.vl)
        ib.add_ops([src_off_chunk, load_op, dst_off_chunk, store])

    ib.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn

def build_pack_RB_v_else(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8):
    fn, (M, Mc, mc, kc, ldM) = _make_fn(f"pack_RB_v_else_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]
    c0, c1, cEB, cRR = _const(0, idx), _const(1, idx), _const(elem_bytes, idx), _const(RR, idx)
    entry.add_ops([c0, c1, cEB, cRR])

    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cRR.result, [], Region([ob])))
    i = ob.args[0]
    rem = arith.SubiOp(mc, i)
    rr = arith.MinSIOp(rem.result, cRR.result)
    bstr = arith.MuliOp(ldM, cEB.result)
    ob.add_ops([rem, rr, bstr])

    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, kc, c1.result, [], Region([ib])))
    j = ib.args[0]

    i_ldM = arith.MuliOp(i, ldM)
    src_off_base = arith.AddiOp(i_ldM.result, j)
    i_kc, j_RR = arith.MuliOp(i, kc), arith.MuliOp(j, cRR.result)
    dst_off_base = arith.AddiOp(i_kc.result, j_RR.result)
    ib.add_ops([i_ldM, src_off_base, i_kc, j_RR, dst_off_base])

    num_vecs = (RR + vlen - 1) // vlen
    for v in range(num_vecs):
        c_chunk, vl_op = get_safe_chunk_avl(rr.result, v, vlen, idx, ib)
        chunk_ldM = arith.MuliOp(c_chunk.result, ldM)
        src_off_chunk = arith.AddiOp(src_off_base.result, chunk_ldM.result)
        load_op = vlse32_v_f32m1Op(M, src_off_chunk.result, bstr.result, vl_op.vl)
        dst_off_chunk = arith.AddiOp(dst_off_base.result, c_chunk.result)
        store = vse32_v_f32m1Op(Mc, dst_off_chunk.result, load_op.result, vl_op.vl)
        ib.add_ops([chunk_ldM, src_off_chunk, load_op, dst_off_chunk, store])

    ib.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn

def build_pack_CB_module(idx: IndexType, memref_type, RR: int = 8, elem_bytes: int = 4, suffix: str = "fp32", vlen: int = 8) -> ModuleOp:
    return ModuleOp([
        build_pack_CB_v_if(idx, memref_type, RR, elem_bytes, suffix, vlen),
        build_pack_CB_v_else(idx, memref_type, RR, elem_bytes, suffix, vlen),
    ])

def build_pack_module(idx: IndexType, memref_type, NR: int = 8, MR: int = 4, elem_bytes: int = 4, suffix: str = "fp32", vlen: int = 8) -> ModuleOp:
    return ModuleOp([
        build_pack_CB_v_if(idx, memref_type, NR, elem_bytes, suffix, vlen),
        build_pack_CB_v_else(idx, memref_type, NR, elem_bytes, suffix, vlen),
        build_pack_RB_v_if(idx, memref_type, MR, elem_bytes, suffix, vlen),
        build_pack_RB_v_else(idx, memref_type, MR, elem_bytes, suffix, vlen),
    ])
