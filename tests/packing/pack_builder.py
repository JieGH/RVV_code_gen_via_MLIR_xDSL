from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.ir import Block, Region

from xdsltemplate.dialects.rvv import (
    SetvlE8M1Op,
    SetvlM2Op,
    SetvlOp,
    compute_lmul,
    vle8_v_i8m1Op,
    vle32_v_f32m1Op,
    vle32_v_f32m2Op,
    vlse8_v_i8m1Op,
    vlse32_v_f32m1Op,
    vlse32_v_f32m2Op,
    vse8_v_i8m1Op,
    vse32_v_f32m1Op,
    vse32_v_f32m2Op,
    vsse8_v_i8m1Op,
)


def _lmul_ops(RR: int, elem_bytes: int, vlen_bits: int = 256):
    if elem_bytes == 1:
        return SetvlE8M1Op, vle8_v_i8m1Op, vlse8_v_i8m1Op, vse8_v_i8m1Op

    lmul = compute_lmul(RR, elem_bytes, vlen_bits)
    if lmul == 1:
        return SetvlOp, vle32_v_f32m1Op, vlse32_v_f32m1Op, vse32_v_f32m1Op
    elif lmul == 2:
        return SetvlM2Op, vle32_v_f32m2Op, vlse32_v_f32m2Op, vse32_v_f32m2Op
    else:
        raise NotImplementedError(f"LMUL={lmul} not implemented.")


def _const(val: int, idx: IndexType) -> arith.ConstantOp:
    return arith.ConstantOp(IntegerAttr(val, idx))


def _make_fn(name: str, memref_type, idx: IndexType):
    fn = func.FuncOp(name, ([memref_type, memref_type, idx, idx, idx], []))
    return fn, fn.args


# FP32 Builders...
def build_pack_CB_v_if(
    idx, memref_type, RR, elem_bytes, suffix, vlen_bits=256, method="AXPY_FP32_Pack"
):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(
        f"pack_CB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx
    )
    entry = fn.body.blocks[0]
    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cEB = _const(elem_bytes, idx)
    cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cEB, cRR])
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cRR.result, [], Region([ob])))
    j = ob.args[0]
    rem = arith.SubiOp(nc, j)
    nr = arith.MinSIOp(rem.result, cRR.result)
    SetvlCls, _, LoadStridedCls, StoreCls = _lmul_ops(RR, elem_bytes, vlen_bits)
    vl_op = SetvlCls(nr.result)
    bstr = arith.MuliOp(ldM, cEB.result)
    ob.add_ops([rem, nr, vl_op, bstr])
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, mc, c1.result, [], Region([ib])))
    i = ib.args[0]
    j_ldM = arith.MuliOp(j, ldM)
    src_off = arith.AddiOp(j_ldM.result, i)
    load_op = LoadStridedCls(M, src_off.result, bstr.result, vl_op.vl)
    j_mc = arith.MuliOp(j, mc)
    i_RR = arith.MuliOp(i, cRR.result)
    dst_off = arith.AddiOp(j_mc.result, i_RR.result)
    store = StoreCls(Mc, dst_off.result, load_op.result, vl_op.vl)
    ib.add_ops([j_ldM, src_off, load_op, j_mc, i_RR, dst_off, store, scf.YieldOp()])
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


def build_pack_CB_v_else(
    idx, memref_type, RR, elem_bytes, suffix, vlen_bits=256, method="AXPY_FP32_Pack"
):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(
        f"pack_CB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx
    )
    entry = fn.body.blocks[0]
    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cRR])
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cRR.result, [], Region([ob])))
    j = ob.args[0]
    rem = arith.SubiOp(nc, j)
    nr = arith.MinSIOp(rem.result, cRR.result)
    SetvlCls, LoadUnitCls, _, StoreCls = _lmul_ops(RR, elem_bytes, vlen_bits)
    vl_op = SetvlCls(nr.result)
    ob.add_ops([rem, nr, vl_op])
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, mc, c1.result, [], Region([ib])))
    i = ib.args[0]
    i_ldM = arith.MuliOp(i, ldM)
    src_off = arith.AddiOp(i_ldM.result, j)
    load_op = LoadUnitCls(M, src_off.result, vl_op.vl)
    j_mc = arith.MuliOp(j, mc)
    i_RR = arith.MuliOp(i, cRR.result)
    dst_off = arith.AddiOp(j_mc.result, i_RR.result)
    store = StoreCls(
        Mc,
        dst_off.result,
        load_op.vl if hasattr(load_op, "vl") else load_op.result,
        vl_op.vl,
    )
    ib.add_ops([i_ldM, src_off, load_op, j_mc, i_RR, dst_off, store, scf.YieldOp()])
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


def build_pack_RB_v_if(
    idx, memref_type, RR, elem_bytes, suffix, vlen_bits=256, method="AXPY_FP32_Pack"
):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(
        f"pack_RB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx
    )
    entry = fn.body.blocks[0]
    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cRR])
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cRR.result, [], Region([ob])))
    i = ob.args[0]
    rem = arith.SubiOp(mc, i)
    rr = arith.MinSIOp(rem.result, cRR.result)
    SetvlCls, LoadUnitCls, _, StoreCls = _lmul_ops(RR, elem_bytes, vlen_bits)
    vl_op = SetvlCls(rr.result)
    ob.add_ops([rem, rr, vl_op])
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, nc, c1.result, [], Region([ib])))
    j = ib.args[0]
    j_ldM = arith.MuliOp(j, ldM)
    src_off = arith.AddiOp(j_ldM.result, i)
    load_op = LoadUnitCls(M, src_off.result, vl_op.vl)
    i_nc = arith.MuliOp(i, nc)
    j_RR = arith.MuliOp(j, cRR.result)
    dst_off = arith.AddiOp(i_nc.result, j_RR.result)
    store = StoreCls(
        Mc,
        dst_off.result,
        load_op.vl if hasattr(load_op, "vl") else load_op.result,
        vl_op.vl,
    )
    ib.add_ops([j_ldM, src_off, load_op, i_nc, j_RR, dst_off, store, scf.YieldOp()])
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


def build_pack_RB_v_else(
    idx, memref_type, RR, elem_bytes, suffix, vlen_bits=256, method="AXPY_FP32_Pack"
):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(
        f"pack_RB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx
    )
    entry = fn.body.blocks[0]
    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cEB = _const(elem_bytes, idx)
    cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cEB, cRR])
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cRR.result, [], Region([ob])))
    i = ob.args[0]
    rem = arith.SubiOp(mc, i)
    rr = arith.MinSIOp(rem.result, cRR.result)
    SetvlCls, _, LoadStridedCls, StoreCls = _lmul_ops(RR, elem_bytes, vlen_bits)
    vl_op = SetvlCls(rr.result)
    bstr = arith.MuliOp(ldM, cEB.result)
    ob.add_ops([rem, rr, vl_op, bstr])
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, nc, c1.result, [], Region([ib])))
    j = ib.args[0]
    i_ldM = arith.MuliOp(i, ldM)
    src_off = arith.AddiOp(i_ldM.result, j)
    load_op = LoadStridedCls(M, src_off.result, bstr.result, vl_op.vl)
    i_nc = arith.MuliOp(i, nc)
    j_RR = arith.MuliOp(j, cRR.result)
    dst_off = arith.AddiOp(i_nc.result, j_RR.result)
    store = StoreCls(Mc, dst_off.result, load_op.result, vl_op.vl)
    ib.add_ops([i_ldM, src_off, load_op, i_nc, j_RR, dst_off, store, scf.YieldOp()])
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


# ==============================================================================
# DOT_INT8_LANE_Pack Builders
# ==============================================================================


def build_pack_RB_dot_int8_lane_if(
    idx, memref_type, RR, elem_bytes, suffix, vlen_bits=256, method="DOT_INT8_LANE_Pack"
):
    fn, (M, Mc, mc, nc, ldM) = _make_fn(
        f"pack_RB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx
    )
    entry = fn.body.blocks[0]

    KR = 32
    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cMR = _const(RR, idx)
    cKR = _const(KR, idx)
    cEB = _const(elem_bytes, idx)
    entry.add_ops([c0, c1, cMR, cKR, cEB])

    # Outer: i = 0..mc step MR
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cMR.result, [], Region([ob])))
    i_out = ob.args[0]
    rem_mr = arith.SubiOp(mc, i_out)
    mr_val = arith.MinSIOp(rem_mr.result, cMR.result)
    ob.add_ops([rem_mr, mr_val])

    # Middle: j = 0..nc step KR (nc is actually kc in packing A)
    mb = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, nc, cKR.result, [], Region([mb])))
    j_mid = mb.args[0]
    rem_kr = arith.SubiOp(nc, j_mid)
    kr_lim = arith.MinSIOp(rem_kr.result, cKR.result)

    SetvlE8M1, _, _, _ = _lmul_ops(1, 1)  # just to get classes
    vl_op = SetvlE8M1(kr_lim.result)
    bstr = arith.MuliOp(ldM, cEB.result)
    mb.add_ops([rem_kr, kr_lim, vl_op, bstr])

    # Inner: ii = 0..mr step 1
    ib = Block(arg_types=[idx])
    mb.add_op(scf.ForOp(c0.result, mr_val.result, c1.result, [], Region([ib])))
    ii_in = ib.args[0]

    # src_off = j_mid * ldM + (i_out + ii_in)
    i_plus_ii = arith.AddiOp(i_out, ii_in)
    j_ldM = arith.MuliOp(j_mid, ldM)
    src_off = arith.AddiOp(j_ldM.result, i_plus_ii.result)

    load_op = vlse8_v_i8m1Op(M, src_off.result, bstr.result, vl_op.vl)

    # dst_off = (i_out / MR) * nc_padded * MR + j_mid * MR + ii_in * KR
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

    ib.add_ops(
        [
            i_plus_ii,
            j_ldM,
            src_off,
            load_op,
            cKR_minus_1,
            nc_plus,
            nc_div,
            nc_padded,
            i_block,
            block_size,
            i_offset,
            j_MR,
            ii_KR,
            t1,
            dst_off,
            store_op,
            scf.YieldOp(),
        ]
    )
    mb.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


def build_pack_CB_dot_int8_lane_if(
    idx, memref_type, RR, elem_bytes, suffix, vlen_bits=256, method="DOT_INT8_LANE_Pack"
):
    """B-side (column-block) packing for DOT_INT8_LANE.

    Uses 4 vlse8 strided loads + 4 vsse8 strided stores (stride=4) to
    produce the interleaved layout without any i32 cast.

    Output layout (for each jj group inside a (j_out, i_mid) tile):
      [v0[0],v1[0],v2[0],v3[0],  v0[1],v1[1],v2[1],v3[1],  ...]
    where vk holds element (K row = i_mid+jj+k, N col = j_out+0..nr-1).
    """
    fn, (M, Mc, mc, nc, ldM) = _make_fn(
        f"pack_CB_v_if_{method}_{suffix}_RR{RR}", memref_type, idx
    )
    entry = fn.body.blocks[0]

    KR = 32
    c0 = _const(0, idx)
    c1 = _const(1, idx)
    c4 = _const(4, idx)  # inner step AND store stride
    cNR = _const(RR, idx)
    cKR = _const(KR, idx)
    entry.add_ops([c0, c1, c4, cNR, cKR])

    # ── Outer loop: j_out = 0..nc step NR  (N-block column index) ─────────
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cNR.result, [], Region([ob])))
    j_out = ob.args[0]
    rem_nr = arith.SubiOp(nc, j_out)
    nr_val = arith.MinSIOp(rem_nr.result, cNR.result)
    ob.add_ops([rem_nr, nr_val])

    # ── Middle loop: i_mid = 0..mc step KR  (K-block row index) ───────────
    mb = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, mc, cKR.result, [], Region([mb])))
    i_mid = mb.args[0]
    rem_kr = arith.SubiOp(mc, i_mid)
    kr_lim = arith.MinSIOp(rem_kr.result, cKR.result)

    # vl = nr_val  (one element per N-column per vector)
    vl_op = SetvlE8M1Op(nr_val.result)
    # byte stride between columns (for vlse8 and vsse8 source stride)
    bstr = arith.MuliOp(ldM, c1.result)  # ldM * 1 = ldM bytes
    mb.add_ops([rem_kr, kr_lim, vl_op, bstr])

    # ── Inner loop: jj = 0..(kr_lim - 4 + 1) step 4, i.e. full groups ────
    #    Guard: kr_lim_div4 * 4 ensures we only process complete groups of 4.
    #    Using integer division: loop bound = (kr_lim // 4) * 4.
    #    Equivalent to: jj < kr_lim - 3, guarded so jj+3 < kr_lim always.
    #    We express as: bound = kr_lim & ~3  (round down to multiple of 4)
    #    MLIR/index: bound = (kr_lim >> 2) << 2 — but xDSL may lack shifts.
    #    Simplest: just use (kr_lim // 4) * 4 via MinSIOp of kr_lim with
    #    itself clamped.  Actually easiest: use step=4 and the for loop
    #    naturally won't run if kr_lim < 4 because (kr_lim - 3) is negative.
    #    Since index is unsigned we clamp with MinSIOp(kr_lim, 4)-based trick:
    #    loop bound = (kr_lim // 4) * 4.  xDSL has DivSI / MulI.
    #    For simplicity use scf.for with bound = kr_lim and step = 4,
    #    but only when kr_lim >= 4.  We skip the sub-3 and use a trick:
    #    bound = kr_lim  (the for loop with step=4 will correctly stop at
    #    the last complete group because jj+4 > kr_lim means no more iters).
    #    *** Wait: scf.for runs while jj < bound. With step=4 starting at 0,
    #    the last iter has jj = largest multiple of 4 < bound.  So bound=kr_lim
    #    is SAFE — jj will be 0,4,8,...,floor((kr_lim-1)/4)*4 and we always
    #    have jj+3 < jj+4 <= kr_lim (not necessarily!).
    #    *** Actually no: if kr_lim=5 and step=4, iters are jj=0 (jj+3=3 ok)
    #    and jj=4 (jj+3=7, but kr_lim=5 → OVERFLOW on reads jj+1..3).
    #    Correct bound: (kr_lim // 4) * 4 using arith.DivSIOp + MuliOp.
    kr_groups = arith.DivSIOp(kr_lim.result, c4.result)  # kr_lim // 4
    kr_loop_bound = arith.MuliOp(kr_groups.result, c4.result)  # * 4
    mb.add_ops([kr_groups, kr_loop_bound])

    ib = Block(arg_types=[idx])
    mb.add_op(scf.ForOp(c0.result, kr_loop_bound.result, c4.result, [], Region([ib])))
    jj_in = ib.args[0]

    # base source offset: M[j_out*ldM + i_mid + jj_in]
    j_ldM = arith.MuliOp(j_out, ldM)
    i_jj = arith.AddiOp(i_mid, jj_in)
    src_base = arith.AddiOp(j_ldM.result, i_jj.result)

    # offsets for the 4 K-rows: src_base+0, +1, +2, +3
    c1b = _const(1, idx)
    c2 = _const(2, idx)
    c3 = _const(3, idx)
    src1 = arith.AddiOp(src_base.result, c1b.result)
    src2 = arith.AddiOp(src_base.result, c2.result)
    src3 = arith.AddiOp(src_base.result, c3.result)

    # 4 strided loads: each collects 'nr' int8 values from consecutive columns
    # vlse8(&M[src_base+k], stride=ldM, vl=nr)
    v0 = vlse8_v_i8m1Op(M, src_base.result, bstr.result, vl_op.vl)
    v1 = vlse8_v_i8m1Op(M, src1.result, bstr.result, vl_op.vl)
    v2 = vlse8_v_i8m1Op(M, src2.result, bstr.result, vl_op.vl)
    v3 = vlse8_v_i8m1Op(M, src3.result, bstr.result, vl_op.vl)

    # destination base: (j_out / NR) * mc_padded * NR + i_mid * NR + jj_in * NR
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

    # dst offsets for the 4 interleaved stores
    dst1 = arith.AddiOp(dst_base.result, c1b.result)
    dst2 = arith.AddiOp(dst_base.result, c2.result)
    dst3 = arith.AddiOp(dst_base.result, c3.result)

    # 4 strided stores with stride=4: interleaves v0..v3 into Mc
    # vsse8(&Mc[dst_base+k], bstride=4, vk, vl)
    # produces layout: [v0[0],v1[0],v2[0],v3[0], v0[1],v1[1],v2[1],v3[1], ...]
    s0 = vsse8_v_i8m1Op(Mc, dst_base.result, c4.result, v0.result, vl_op.vl)
    s1 = vsse8_v_i8m1Op(Mc, dst1.result, c4.result, v1.result, vl_op.vl)
    s2 = vsse8_v_i8m1Op(Mc, dst2.result, c4.result, v2.result, vl_op.vl)
    s3 = vsse8_v_i8m1Op(Mc, dst3.result, c4.result, v3.result, vl_op.vl)

    ib.add_ops(
        [
            j_ldM,
            i_jj,
            src_base,
            c1b,
            c2,
            c3,
            src1,
            src2,
            src3,
            v0,
            v1,
            v2,
            v3,
            cKR_minus_1,
            mc_plus,
            mc_div,
            mc_padded,
            j_block,
            block_size,
            j_offset,
            i_NR,
            jj_NR,
            t1,
            dst_base,
            dst1,
            dst2,
            dst3,
            s0,
            s1,
            s2,
            s3,
            scf.YieldOp(),
        ]
    )
    mb.add_op(scf.YieldOp())
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


def build_pack_RB_dot_int8_lane_else(
    idx, memref_type, RR, elem_bytes, suffix, vlen_bits=256, method="DOT_INT8_LANE_Pack"
):
    fn, args = _make_fn(f"pack_RB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx)
    fn.body.blocks[0].add_op(func.ReturnOp())
    return fn


def build_pack_CB_dot_int8_lane_else(
    idx, memref_type, RR, elem_bytes, suffix, vlen_bits=256, method="DOT_INT8_LANE_Pack"
):
    fn, args = _make_fn(f"pack_CB_v_else_{method}_{suffix}_RR{RR}", memref_type, idx)
    fn.body.blocks[0].add_op(func.ReturnOp())
    return fn


def build_pack_module(
    idx: IndexType,
    memref_type,
    NR: int = 8,
    MR: int = 8,
    elem_bytes: int = 4,
    suffix: str = "fp32",
    vlen_bits: int = 256,
    method: str = "AXPY_FP32_Pack",
) -> ModuleOp:
    if method == "DOT_INT8_LANE_Pack":
        return ModuleOp(
            [
                build_pack_CB_dot_int8_lane_if(
                    idx, memref_type, NR, elem_bytes, suffix, vlen_bits, method
                ),
                build_pack_CB_dot_int8_lane_else(
                    idx, memref_type, NR, elem_bytes, suffix, vlen_bits, method
                ),
                build_pack_RB_dot_int8_lane_if(
                    idx, memref_type, MR, elem_bytes, suffix, vlen_bits, method
                ),
                build_pack_RB_dot_int8_lane_else(
                    idx, memref_type, MR, elem_bytes, suffix, vlen_bits, method
                ),
            ]
        )
    return ModuleOp(
        [
            build_pack_CB_v_if(
                idx, memref_type, NR, elem_bytes, suffix, vlen_bits, method
            ),
            build_pack_CB_v_else(
                idx, memref_type, NR, elem_bytes, suffix, vlen_bits, method
            ),
            build_pack_RB_v_if(
                idx, memref_type, MR, elem_bytes, suffix, vlen_bits, method
            ),
            build_pack_RB_v_else(
                idx, memref_type, MR, elem_bytes, suffix, vlen_bits, method
            ),
        ]
    )
