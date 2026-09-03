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
    SetvlE8M1Op,
    vle8_v_i8m1Op,
    vlse8_v_i8m1Op,
    vse8_v_i8m1Op,
    vsse8_v_i8m1Op,
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

def build_pack_CB_v_if(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8, method: str = "AXPY_FP32_Pack"):
    fn, (M, Mc, kc, nc, ldM) = _make_fn(f"pack_CB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx)
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

def build_pack_CB_v_else(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8, method: str = "AXPY_FP32_Pack"):
    fn, (M, Mc, kc, nc, ldM) = _make_fn(f"pack_CB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx)
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

def build_pack_RB_v_if(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8, method: str = "AXPY_FP32_Pack"):
    fn, (M, Mc, mc, kc, ldM) = _make_fn(f"pack_RB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx)
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

def build_pack_RB_v_else(idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8, method: str = "AXPY_FP32_Pack"):
    fn, (M, Mc, mc, kc, ldM) = _make_fn(f"pack_RB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx)
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

def build_pack_RB_dot_int8_lane_if(idx, memref_type, RR, elem_bytes, suffix, vlen=8, method="DOT_INT8_LANE_Pack"):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(f"pack_RB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]
    KR = 32
    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cMR = _const(RR, idx)
    cKR = _const(KR, idx)
    cEB = _const(elem_bytes, idx)
    entry.add_ops([c0, c1, cMR, cKR, cEB])
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cMR.result, [], Region([ob])))
    i_out = ob.args[0]
    rem_mr = arith.SubiOp(mc, i_out)
    mr_val = arith.MinSIOp(rem_mr.result, cMR.result)
    ob.add_ops([rem_mr, mr_val])
    mb = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, nc, cKR.result, [], Region([mb])))
    j_mid = mb.args[0]
    rem_kr = arith.SubiOp(nc, j_mid)
    kr_lim = arith.MinSIOp(rem_kr.result, cKR.result)
    vl_op = SetvlE8M1Op(kr_lim.result)
    bstr = arith.MuliOp(ldM, cEB.result)
    mb.add_ops([rem_kr, kr_lim, vl_op, bstr])
    ib = Block(arg_types=[idx])
    mb.add_op(scf.ForOp(c0.result, mr_val.result, c1.result, [], Region([ib])))
    ii_in = ib.args[0]
    i_plus_ii = arith.AddiOp(i_out, ii_in)
    j_ldM = arith.MuliOp(j_mid, ldM)
    src_off = arith.AddiOp(j_ldM.result, i_plus_ii.result)
    load_op = vlse8_v_i8m1Op(M, src_off.result, bstr.result, vl_op.vl)
    cKR_minus_1 = _const(KR - 1, idx)
    nc_plus = arith.AddiOp(nc, cKR_minus_1.result)
    nc_div = arith.DivSIOp(nc_plus.result, cKR.result)
    nc_padded = arith.MuliOp(nc_div.result, cKR.result)
    i_block = arith.DivSIOp(i_out, cMR.result)
    block_size = arith.MuliOp(nc_padded.result, cMR.result)
    i_offset = arith.MuliOp(i_block.result, block_size.result)
    j_MR = arith.MuliOp(j_mid, cMR.result)
    ii_KR = arith.MuliOp(ii_in, cKR.result)
    t1 = arith.AddiOp(i_offset.result, j_MR.result)
    dst_off = arith.AddiOp(t1.result, ii_KR.result)
    store_op = vse8_v_i8m1Op(Mc, dst_off.result, load_op.result, vl_op.vl)
    ib.add_ops([i_plus_ii, j_ldM, src_off, load_op, cKR_minus_1, nc_plus, nc_div, nc_padded, i_block, block_size, i_offset, j_MR, ii_KR, t1, dst_off, store_op, scf.YieldOp()])
    mb.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn

def build_pack_CB_dot_int8_lane_if(idx, memref_type, RR, elem_bytes, suffix, vlen=8, method="DOT_INT8_LANE_Pack"):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(f"pack_CB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]
    KR = 32
    c0 = _const(0, idx)
    c1 = _const(1, idx)
    c4 = _const(4, idx)
    cNR = _const(RR, idx)
    cKR = _const(KR, idx)
    entry.add_ops([c0, c1, c4, cNR, cKR])
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cNR.result, [], Region([ob])))
    j_out = ob.args[0]
    rem_nr = arith.SubiOp(nc, j_out)
    nr_val = arith.MinSIOp(rem_nr.result, cNR.result)
    ob.add_ops([rem_nr, nr_val])
    mb = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, mc, cKR.result, [], Region([mb])))
    i_mid = mb.args[0]
    rem_kr = arith.SubiOp(mc, i_mid)
    kr_lim = arith.MinSIOp(rem_kr.result, cKR.result)
    vl_op = SetvlE8M1Op(nr_val.result)
    bstr = arith.MuliOp(ldM, c1.result)
    mb.add_ops([rem_kr, kr_lim, vl_op, bstr])
    kr_groups = arith.DivSIOp(kr_lim.result, c4.result)
    kr_loop_bound = arith.MuliOp(kr_groups.result, c4.result)
    mb.add_ops([kr_groups, kr_loop_bound])
    ib = Block(arg_types=[idx])
    mb.add_op(scf.ForOp(c0.result, kr_loop_bound.result, c4.result, [], Region([ib])))
    jj_in = ib.args[0]
    j_ldM = arith.MuliOp(j_out, ldM)
    i_jj = arith.AddiOp(i_mid, jj_in)
    src_base = arith.AddiOp(j_ldM.result, i_jj.result)
    c1b = _const(1, idx)
    c2 = _const(2, idx)
    c3 = _const(3, idx)
    src1 = arith.AddiOp(src_base.result, c1b.result)
    src2 = arith.AddiOp(src_base.result, c2.result)
    src3 = arith.AddiOp(src_base.result, c3.result)
    v0 = vlse8_v_i8m1Op(M, src_base.result, bstr.result, vl_op.vl)
    v1 = vlse8_v_i8m1Op(M, src1.result, bstr.result, vl_op.vl)
    v2 = vlse8_v_i8m1Op(M, src2.result, bstr.result, vl_op.vl)
    v3 = vlse8_v_i8m1Op(M, src3.result, bstr.result, vl_op.vl)
    cKR_minus_1 = _const(KR - 1, idx)
    mc_plus = arith.AddiOp(mc, cKR_minus_1.result)
    mc_div = arith.DivSIOp(mc_plus.result, cKR.result)
    mc_padded = arith.MuliOp(mc_div.result, cKR.result)
    j_block = arith.DivSIOp(j_out, cNR.result)
    block_size = arith.MuliOp(mc_padded.result, cNR.result)
    j_offset = arith.MuliOp(j_block.result, block_size.result)
    i_NR = arith.MuliOp(i_mid, cNR.result)
    jj_NR = arith.MuliOp(jj_in, cNR.result)
    t1 = arith.AddiOp(j_offset.result, i_NR.result)
    dst_base = arith.AddiOp(t1.result, jj_NR.result)
    dst1 = arith.AddiOp(dst_base.result, c1b.result)
    dst2 = arith.AddiOp(dst_base.result, c2.result)
    dst3 = arith.AddiOp(dst_base.result, c3.result)
    s0 = vsse8_v_i8m1Op(Mc, dst_base.result, c4.result, v0.result, vl_op.vl)
    s1 = vsse8_v_i8m1Op(Mc, dst1.result, c4.result, v1.result, vl_op.vl)
    s2 = vsse8_v_i8m1Op(Mc, dst2.result, c4.result, v2.result, vl_op.vl)
    s3 = vsse8_v_i8m1Op(Mc, dst3.result, c4.result, v3.result, vl_op.vl)
    ib.add_ops([j_ldM, i_jj, src_base, c1b, c2, c3, src1, src2, src3, v0, v1, v2, v3, cKR_minus_1, mc_plus, mc_div, mc_padded, j_block, block_size, j_offset, i_NR, jj_NR, t1, dst_base, dst1, dst2, dst3, s0, s1, s2, s3, scf.YieldOp()])
    mb.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn

def build_pack_RB_dot_int8_lane_else(idx, memref_type, RR, elem_bytes, suffix, vlen=8, method="DOT_INT8_LANE_Pack"):
    fn, args = _make_fn(f"pack_RB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx)
    fn.body.blocks[0].add_op(func.ReturnOp())
    return fn

def build_pack_CB_dot_int8_lane_else(idx, memref_type, RR, elem_bytes, suffix, vlen=8, method="DOT_INT8_LANE_Pack"):
    fn, args = _make_fn(f"pack_CB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx)
    fn.body.blocks[0].add_op(func.ReturnOp())
    return fn

def build_pack_RB_dot_fp32_lane_if(idx, memref_type, RR, elem_bytes, suffix, vlen=8, method="DOT_FP32_LANE_Pack"):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(f"pack_RB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]
    KR = 4
    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cMR = _const(RR, idx)
    cKR = _const(KR, idx)
    cEB = _const(elem_bytes, idx)
    entry.add_ops([c0, c1, cMR, cKR, cEB])
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cMR.result, [], Region([ob])))
    i_out = ob.args[0]
    rem_mr = arith.SubiOp(mc, i_out)
    mr_val = arith.MinSIOp(rem_mr.result, cMR.result)
    ob.add_ops([rem_mr, mr_val])
    mb = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, nc, cKR.result, [], Region([mb])))
    j_mid = mb.args[0]
    rem_kr = arith.SubiOp(nc, j_mid)
    kr_lim = arith.MinSIOp(rem_kr.result, cKR.result)
    vl_op = SetvlOp(kr_lim.result)
    bstr = arith.MuliOp(ldM, cEB.result)
    mb.add_ops([rem_kr, kr_lim, vl_op, bstr])
    ib = Block(arg_types=[idx])
    mb.add_op(scf.ForOp(c0.result, mr_val.result, c1.result, [], Region([ib])))
    ii_in = ib.args[0]
    i_plus_ii = arith.AddiOp(i_out, ii_in)
    j_ldM = arith.MuliOp(j_mid, ldM)
    src_off = arith.AddiOp(j_ldM.result, i_plus_ii.result)
    load_op = vlse32_v_f32m1Op(M, src_off.result, bstr.result, vl_op.vl)
    cKR_minus_1 = _const(KR - 1, idx)
    nc_plus = arith.AddiOp(nc, cKR_minus_1.result)
    nc_div = arith.DivSIOp(nc_plus.result, cKR.result)
    nc_padded = arith.MuliOp(nc_div.result, cKR.result)
    i_block = arith.DivSIOp(i_out, cMR.result)
    block_size = arith.MuliOp(nc_padded.result, cMR.result)
    i_offset = arith.MuliOp(i_block.result, block_size.result)
    j_MR = arith.MuliOp(j_mid, cMR.result)
    ii_KR = arith.MuliOp(ii_in, cKR.result)
    t1 = arith.AddiOp(i_offset.result, j_MR.result)
    dst_off = arith.AddiOp(t1.result, ii_KR.result)
    store_op = vse32_v_f32m1Op(Mc, dst_off.result, load_op.result, vl_op.vl)
    ib.add_ops([i_plus_ii, j_ldM, src_off, load_op, cKR_minus_1, nc_plus, nc_div, nc_padded, i_block, block_size, i_offset, j_MR, ii_KR, t1, dst_off, store_op, scf.YieldOp()])
    mb.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn

def build_pack_CB_dot_fp32_lane_if(idx, memref_type, RR, elem_bytes, suffix, vlen=8, method="DOT_FP32_LANE_Pack"):
    fn, (M, Mc, kc, nc, ldM) = _make_fn(f"pack_CB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]
    KR = 4
    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cNR = _const(RR, idx)
    cKR = _const(KR, idx)
    entry.add_ops([c0, c1, cNR, cKR])
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cNR.result, [], Region([ob])))
    j_out = ob.args[0]
    rem_nr = arith.SubiOp(nc, j_out)
    nr_val = arith.MinSIOp(rem_nr.result, cNR.result)
    ob.add_ops([rem_nr, nr_val])
    mb = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, kc, cKR.result, [], Region([mb])))
    i_mid = mb.args[0]
    rem_kr = arith.SubiOp(kc, i_mid)
    kr_lim = arith.MinSIOp(rem_kr.result, cKR.result)
    vl_op = SetvlOp(kr_lim.result)
    mb.add_ops([rem_kr, kr_lim, vl_op])
    ib = Block(arg_types=[idx])
    mb.add_op(scf.ForOp(c0.result, nr_val.result, c1.result, [], Region([ib])))
    ii_in = ib.args[0]
    j_plus_ii = arith.AddiOp(j_out, ii_in)
    j_ldM = arith.MuliOp(j_plus_ii.result, ldM)
    src_off = arith.AddiOp(j_ldM.result, i_mid)
    load_op = vle32_v_f32m1Op(M, src_off.result, vl_op.vl)
    cKR_minus_1 = _const(KR - 1, idx)
    kc_plus = arith.AddiOp(kc, cKR_minus_1.result)
    kc_div = arith.DivSIOp(kc_plus.result, cKR.result)
    kc_padded = arith.MuliOp(kc_div.result, cKR.result)
    j_block = arith.DivSIOp(j_out, cNR.result)
    block_size = arith.MuliOp(kc_padded.result, cNR.result)
    j_offset = arith.MuliOp(j_block.result, block_size.result)
    i_mid_div_KR = arith.DivSIOp(i_mid, cKR.result)
    NR_KR = arith.MuliOp(cNR.result, cKR.result)
    i_offset = arith.MuliOp(i_mid_div_KR.result, NR_KR.result)
    ii_KR = arith.MuliOp(ii_in, cKR.result)
    t1 = arith.AddiOp(j_offset.result, i_offset.result)
    dst_off = arith.AddiOp(t1.result, ii_KR.result)
    store_op = vse32_v_f32m1Op(Mc, dst_off.result, load_op.results[0], vl_op.vl)
    ib.add_ops([j_plus_ii, j_ldM, src_off, load_op, cKR_minus_1, kc_plus, kc_div, kc_padded, j_block, block_size, j_offset, i_mid_div_KR, NR_KR, i_offset, ii_KR, t1, dst_off, store_op, scf.YieldOp()])
    mb.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn

def build_pack_RB_dot_fp32_lane_else(idx, memref_type, RR, elem_bytes, suffix, vlen=8, method="DOT_FP32_LANE_Pack"):
    fn, args = _make_fn(f"pack_RB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx)
    fn.body.blocks[0].add_op(func.ReturnOp())
    return fn

def build_pack_CB_dot_fp32_lane_else(idx, memref_type, RR, elem_bytes, suffix, vlen=8, method="DOT_FP32_LANE_Pack"):
    fn, args = _make_fn(f"pack_CB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx)
    fn.body.blocks[0].add_op(func.ReturnOp())
    return fn

def build_pack_CB_v_if_int8(idx: IndexType, memref_type, RR: int, elem_bytes: int,
                              suffix: str, vlen: int = 32, method: str = "AXPY_INT8_Pack"):
    """
    Pack Matrix B (CB = Column-Major B -> packed Bc row-major) for INT8 AXPY.
    Uses vle8/vlse8/vse8 with LMUL=M1. Mirrors build_pack_CB_v_if but for 8-bit.

    Arguments passed at runtime: (M, Mc, kc, nc, ldM)
    Loop: outer over j (nc, step NR), inner over i (kc, step 1).
    Load: vlse8 with stride = ldM * 1 (gathers NR elements across rows at same k).
    Store: vse8 contiguous to Bc.
    """
    fn, (M, Mc, kc, nc, ldM) = _make_fn(f"pack_CB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx)
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

    # Inline chunk loop using SetvlE8M1Op (do NOT use get_safe_chunk_avl which calls FP32 SetvlOp)
    num_vecs = (RR + vlen - 1) // vlen
    for v in range(num_vecs):
        c_chunk     = _const(v * vlen, idx)
        c_chunk_end = _const((v + 1) * vlen, idx)
        end         = arith.MinSIOp(nr.result, c_chunk_end.result)
        end_safe    = arith.MaxSIOp(end.result, c_chunk.result)
        chunk_avl   = arith.SubiOp(end_safe.result, c_chunk.result)
        vl_op       = SetvlE8M1Op(chunk_avl.result)   # <- INT8 M1 setvl
        ib.add_ops([c_chunk, c_chunk_end, end, end_safe, chunk_avl, vl_op])

        chunk_ldM     = arith.MuliOp(c_chunk.result, ldM)
        src_off_chunk = arith.AddiOp(src_off_base.result, chunk_ldM.result)
        load_op       = vlse8_v_i8m1Op(M, src_off_chunk.result, bstr.result, vl_op.vl)
        dst_off_chunk = arith.AddiOp(dst_off_base.result, c_chunk.result)
        store         = vse8_v_i8m1Op(Mc, dst_off_chunk.result, load_op.result, vl_op.vl)
        ib.add_ops([chunk_ldM, src_off_chunk, load_op, dst_off_chunk, store])

    ib.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn

def build_pack_CB_v_else_int8(idx: IndexType, memref_type, RR: int, elem_bytes: int,
                               suffix: str, vlen: int = 32, method: str = "AXPY_INT8_Pack"):
    """Stub -- INT8 CB else branch. Tail cases handled by _if via SetvlE8M1 masking."""
    fn, _ = _make_fn(f"pack_CB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx)
    fn.body.blocks[0].add_op(func.ReturnOp())
    return fn

def build_pack_RB_v_if_int8(idx: IndexType, memref_type, RR: int, elem_bytes: int,
                              suffix: str, vlen: int = 32, method: str = "AXPY_INT8_Pack"):
    """
    Pack Matrix A (RB = Row-Major B -> packed Ac column-major panels) for INT8 AXPY.
    Uses vle8/vse8 with LMUL=M1. Mirrors build_pack_RB_v_if but for 8-bit.

    Arguments passed at runtime: (M, Mc, mc, kc, ldM)
    Loop: outer over i (mc, step MR), inner over j (kc, step 1).
    Load: vle8 contiguous (MR elements at same k, adjacent rows in col-major A).
    Store: vse8 contiguous to Ac.
    """
    fn, (M, Mc, mc, kc, ldM) = _make_fn(f"pack_RB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx)
    entry = fn.body.blocks[0]
    c0, c1, cRR = _const(0, idx), _const(1, idx), _const(RR, idx)
    entry.add_ops([c0, c1, cRR])

    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cRR.result, [], Region([ob])))
    i = ob.args[0]
    rem = arith.SubiOp(mc, i)
    rr  = arith.MinSIOp(rem.result, cRR.result)
    ob.add_ops([rem, rr])

    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, kc, c1.result, [], Region([ib])))
    j = ib.args[0]

    j_ldM        = arith.MuliOp(j, ldM)
    src_off_base = arith.AddiOp(j_ldM.result, i)
    i_kc, j_RR  = arith.MuliOp(i, kc), arith.MuliOp(j, cRR.result)
    dst_off_base = arith.AddiOp(i_kc.result, j_RR.result)
    ib.add_ops([j_ldM, src_off_base, i_kc, j_RR, dst_off_base])

    num_vecs = (RR + vlen - 1) // vlen
    for v in range(num_vecs):
        c_chunk     = _const(v * vlen, idx)
        c_chunk_end = _const((v + 1) * vlen, idx)
        end         = arith.MinSIOp(rr.result, c_chunk_end.result)
        end_safe    = arith.MaxSIOp(end.result, c_chunk.result)
        chunk_avl   = arith.SubiOp(end_safe.result, c_chunk.result)
        vl_op       = SetvlE8M1Op(chunk_avl.result)   # <- INT8 M1 setvl
        ib.add_ops([c_chunk, c_chunk_end, end, end_safe, chunk_avl, vl_op])

        src_off_chunk = arith.AddiOp(src_off_base.result, c_chunk.result)
        load_op       = vle8_v_i8m1Op(M, src_off_chunk.result, vl_op.vl)
        dst_off_chunk = arith.AddiOp(dst_off_base.result, c_chunk.result)
        store         = vse8_v_i8m1Op(Mc, dst_off_chunk.result, load_op.result, vl_op.vl)
        ib.add_ops([src_off_chunk, load_op, dst_off_chunk, store])

    ib.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn

def build_pack_RB_v_else_int8(idx: IndexType, memref_type, RR: int, elem_bytes: int,
                               suffix: str, vlen: int = 32, method: str = "AXPY_INT8_Pack"):
    """Stub -- INT8 RB else branch. Tail cases handled by _if via SetvlE8M1 masking."""
    fn, _ = _make_fn(f"pack_RB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx)
    fn.body.blocks[0].add_op(func.ReturnOp())
    return fn

def build_pack_module(idx: IndexType, memref_type, NR: int = 8, MR: int = 4, elem_bytes: int = 4, suffix: str = "fp32", vlen: int = 8, method: str = "AXPY_FP32_Pack") -> ModuleOp:
    if method == "AXPY_INT8_Pack":
        return ModuleOp([
            build_pack_CB_v_if_int8(idx, memref_type, NR, elem_bytes, suffix, vlen, method),
            build_pack_CB_v_else_int8(idx, memref_type, NR, elem_bytes, suffix, vlen, method),
            build_pack_RB_v_if_int8(idx, memref_type, MR, elem_bytes, suffix, vlen, method),
            build_pack_RB_v_else_int8(idx, memref_type, MR, elem_bytes, suffix, vlen, method),
        ])
    if method == "DOT_INT8_LANE_Pack":
        return ModuleOp([
            build_pack_CB_dot_int8_lane_if(idx, memref_type, NR, elem_bytes, suffix, vlen, method),
            build_pack_CB_dot_int8_lane_else(idx, memref_type, NR, elem_bytes, suffix, vlen, method),
            build_pack_RB_dot_int8_lane_if(idx, memref_type, MR, elem_bytes, suffix, vlen, method),
            build_pack_RB_dot_int8_lane_else(idx, memref_type, MR, elem_bytes, suffix, vlen, method),
        ])
    if method == "DOT_FP32_LANE_Pack":
        return ModuleOp([
            build_pack_CB_dot_fp32_lane_if(idx, memref_type, NR, elem_bytes, suffix, vlen, method),
            build_pack_CB_dot_fp32_lane_else(idx, memref_type, NR, elem_bytes, suffix, vlen, method),
            build_pack_RB_dot_fp32_lane_if(idx, memref_type, MR, elem_bytes, suffix, vlen, method),
            build_pack_RB_dot_fp32_lane_else(idx, memref_type, MR, elem_bytes, suffix, vlen, method),
        ])
    return ModuleOp([
        build_pack_CB_v_if(idx, memref_type, NR, elem_bytes, suffix, vlen, method),
        build_pack_CB_v_else(idx, memref_type, NR, elem_bytes, suffix, vlen, method),
        build_pack_RB_v_if(idx, memref_type, MR, elem_bytes, suffix, vlen, method),
        build_pack_RB_v_else(idx, memref_type, MR, elem_bytes, suffix, vlen, method),
    ])
