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
from xdsl.ir import Block, Region

from xdsltemplate.dialects.rvv import (
    SetvlOp,
    vle32_v_f32m1Op,
    vlse32_v_f32m1Op,
    vse32_v_f32m1Op,
)

# ── Internal helpers ────────────────────────────────────────────────────────────


def _const(val: int, idx: IndexType) -> arith.ConstantOp:
    return arith.ConstantOp(IntegerAttr(val, idx))


def _make_fn(name: str, memref_type, idx: IndexType, dim1_name: str = "dim1", dim2_name: str = "dim2"):
    """Create a func.FuncOp with signature (M*, Mc*, dim1, dim2, ldM) -> ().
    
    Matches the BLIS reference signatures:
      pack_CB( orderM, transM, mc/kc, nc, M, ldM, Mc, RR )  -- dim1=kc (rows to pack), dim2=nc
      pack_RB( orderM, transM, mc,    kc, M, ldM, Mc, RR )  -- dim1=mc, dim2=kc (cols/K to pack)
    """
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
def build_pack_CB_v_if(
    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str
):
    """
    pack_CB if-branch: vlse32 strided load (Col-Major Non-Trans B).

    Matches reference: pack_CB(orderB, transB, kc, nc, Bptr, ldB, Bc, NR)
      Signature: (B*, Bc*, kc, nc, ldB) -> ()
      kc = K-dimension panel size (rows of B in col-major)
      nc = N-dimension panel size (cols of B)

    Access: Bc[j * kc + i * RR .. + vl] = B[(j+0)*ldB+i, (j+1)*ldB+i, ...]
    """
    fn, (M, Mc, kc, nc, ldM) = _make_fn(
        f"pack_CB_v_if_{suffix}_RR{RR}", memref_type, idx
    )
    entry = fn.body.blocks[0]

    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cEB = _const(elem_bytes, idx)
    cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cEB, cRR])

    # Outer: j = 0..nc step RR  (N-dimension blocks)
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cRR.result, [], Region([ob])))
    j = ob.args[0]

    rem = arith.SubiOp(nc, j)
    nr = arith.MinSIOp(rem.result, cRR.result)    # vl = min(nc-j, NR): guards N boundary
    vl_op = SetvlOp(nr.result)
    bstr = arith.MuliOp(ldM, cEB.result)          # byte stride = ldB * elem_bytes (col-major stride)
    ob.add_ops([rem, nr, vl_op, bstr])

    # Inner: i = 0..kc step 1  (K-dimension elements, i.e. rows of B)
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, kc, c1.result, [], Region([ib])))
    i = ib.args[0]

    # src_off = j * ldB + i  (col-major: column j, row i)
    j_ldM = arith.MuliOp(j, ldM)
    src_off = arith.AddiOp(j_ldM.result, i)
    load_op = vlse32_v_f32m1Op(M, src_off.result, bstr.result, vl_op.vl)

    # dst_off = j * kc + i * RR  (packed panel layout)
    j_kc = arith.MuliOp(j, kc)
    i_RR = arith.MuliOp(i, cRR.result)
    dst_off = arith.AddiOp(j_kc.result, i_RR.result)
    store = vse32_v_f32m1Op(Mc, dst_off.result, load_op.result, vl_op.vl)

    ib.add_ops([j_ldM, src_off, load_op, j_kc, i_RR, dst_off, store, scf.YieldOp()])
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
def build_pack_CB_v_else(
    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str
):
    """
    pack_CB else-branch: vle32 unit-stride load (Row-Major Non-Trans B).

    Matches reference: pack_CB(orderB, transB, kc, nc, Bptr, ldB, Bc, NR)
      Signature: (B*, Bc*, kc, nc, ldB) -> ()
      kc = K-dimension panel size (cols of B in row-major)
      nc = N-dimension panel size (rows of B)

    Access: Bc[j * kc + i * RR .. + vl] = B[i*ldB+j .. i*ldB+j+vl-1] (contiguous)
    """
    fn, (M, Mc, kc, nc, ldM) = _make_fn(
        f"pack_CB_v_else_{suffix}_RR{RR}", memref_type, idx
    )
    entry = fn.body.blocks[0]

    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cRR])

    # Outer: j = 0..nc step RR  (N-dimension blocks)
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, nc, cRR.result, [], Region([ob])))
    j = ob.args[0]

    rem = arith.SubiOp(nc, j)
    nr = arith.MinSIOp(rem.result, cRR.result)    # vl = min(nc-j, NR)
    vl_op = SetvlOp(nr.result)
    ob.add_ops([rem, nr, vl_op])

    # Inner: i = 0..kc step 1  (K-dimension elements)
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, kc, c1.result, [], Region([ib])))
    i = ib.args[0]

    # src_off = i * ldB + j  (row-major: row i, starting at col j)
    i_ldM = arith.MuliOp(i, ldM)
    src_off = arith.AddiOp(i_ldM.result, j)
    load_op = vle32_v_f32m1Op(M, src_off.result, vl_op.vl)

    # dst_off = j * kc + i * RR  (packed panel layout)
    j_kc = arith.MuliOp(j, kc)
    i_RR = arith.MuliOp(i, cRR.result)
    dst_off = arith.AddiOp(j_kc.result, i_RR.result)
    store = vse32_v_f32m1Op(Mc, dst_off.result, load_op.results[0], vl_op.vl)

    ib.add_ops([i_ldM, src_off, load_op, j_kc, i_RR, dst_off, store, scf.YieldOp()])
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
def build_pack_RB_v_if(
    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str
):
    """
    pack_RB if-branch: vle32 unit-stride load (Col-Major Non-Trans A).

    Matches reference: pack_RB(orderA, transA, mc, kc, Aptr, ldA, Ac, MR)
      Signature: (A*, Ac*, mc, kc, ldA) -> ()
      mc = M-dimension panel size (rows of A)
      kc = K-dimension panel size (cols of A in col-major, passed as 'nc' in _make_fn generic)

    NOTE: The generic _make_fn parameter 'nc' corresponds to 'kc' at the call site
    (macro_builder.py calls: pack_RB(Aptr, Ac, mc, kc, ldA)).

    Access: Ac[i * kc + j * RR .. + vl] = A[j*ldA+i .. j*ldA+i+rr-1] (contiguous col-major)
    """
    fn, (M, Mc, mc, kc, ldM) = _make_fn(
        f"pack_RB_v_if_{suffix}_RR{RR}", memref_type, idx
    )
    entry = fn.body.blocks[0]

    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cRR])

    # Outer: i = 0..mc step RR  (M-dimension row blocks)
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cRR.result, [], Region([ob])))
    i = ob.args[0]

    rem = arith.SubiOp(mc, i)
    rr = arith.MinSIOp(rem.result, cRR.result)    # vl = min(mc-i, MR): guards M boundary
    vl_op = SetvlOp(rr.result)
    ob.add_ops([rem, rr, vl_op])

    # Inner: j = 0..kc step 1  (K-dimension elements, all columns of A)
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, kc, c1.result, [], Region([ib])))
    j = ib.args[0]

    # src_off = j * ldA + i  → A[j*ldA+i..j*ldA+i+rr-1] unit-stride (contiguous rows in col-major)
    j_ldM = arith.MuliOp(j, ldM)
    src_off = arith.AddiOp(j_ldM.result, i)
    load_op = vle32_v_f32m1Op(M, src_off.result, vl_op.vl)

    # dst_off = i * kc + j * RR  (packed panel: row-block i, k-column j)
    i_kc = arith.MuliOp(i, kc)
    j_RR = arith.MuliOp(j, cRR.result)
    dst_off = arith.AddiOp(i_kc.result, j_RR.result)
    store = vse32_v_f32m1Op(Mc, dst_off.result, load_op.results[0], vl_op.vl)

    ib.add_ops([j_ldM, src_off, load_op, i_kc, j_RR, dst_off, store, scf.YieldOp()])
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
def build_pack_RB_v_else(
    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str
):
    """
    pack_RB else-branch: vlse32 strided load (Row-Major Non-Trans A).

    Matches reference: pack_RB(orderA, transA, mc, kc, Aptr, ldA, Ac, MR)
      Signature: (A*, Ac*, mc, kc, ldA) -> ()
      mc = M-dimension panel size (rows of A)
      kc = K-dimension panel size (cols of A, passed as 'nc' in _make_fn generic)

    NOTE: The generic _make_fn parameter 'nc' corresponds to 'kc' at the call site.

    Access: Ac[i * kc + j * RR .. + vl] = A[i*ldA+j, (i+1)*ldA+j, ...] (stride=ldA)
    """
    fn, (M, Mc, mc, kc, ldM) = _make_fn(
        f"pack_RB_v_else_{suffix}_RR{RR}", memref_type, idx
    )
    entry = fn.body.blocks[0]

    c0 = _const(0, idx)
    c1 = _const(1, idx)
    cEB = _const(elem_bytes, idx)
    cRR = _const(RR, idx)
    entry.add_ops([c0, c1, cEB, cRR])

    # Outer: i = 0..mc step RR  (M-dimension row blocks)
    ob = Block(arg_types=[idx])
    entry.add_op(scf.ForOp(c0.result, mc, cRR.result, [], Region([ob])))
    i = ob.args[0]

    rem = arith.SubiOp(mc, i)
    rr = arith.MinSIOp(rem.result, cRR.result)    # vl = min(mc-i, MR)
    vl_op = SetvlOp(rr.result)
    bstr = arith.MuliOp(ldM, cEB.result)          # byte stride = ldA * elem_bytes
    ob.add_ops([rem, rr, vl_op, bstr])

    # Inner: j = 0..kc step 1  (K-dimension elements)
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, kc, c1.result, [], Region([ib])))
    j = ib.args[0]

    # src_off = i * ldA + j  → A[(i+0)*ldA+j], A[(i+1)*ldA+j]... stride=ldA
    i_ldM = arith.MuliOp(i, ldM)
    src_off = arith.AddiOp(i_ldM.result, j)
    load_op = vlse32_v_f32m1Op(M, src_off.result, bstr.result, vl_op.vl)

    # dst_off = i * kc + j * RR  (packed panel: row-block i, k-column j)
    i_kc = arith.MuliOp(i, kc)
    j_RR = arith.MuliOp(j, cRR.result)
    dst_off = arith.AddiOp(i_kc.result, j_RR.result)
    store = vse32_v_f32m1Op(Mc, dst_off.result, load_op.result, vl_op.vl)

    ib.add_ops([i_ldM, src_off, load_op, i_kc, j_RR, dst_off, store, scf.YieldOp()])
    ob.add_op(scf.YieldOp())
    entry.add_op(func.ReturnOp())
    return fn


# ── Module builders ────────────────────────────────────────────────────────────


def build_pack_CB_module(
    idx: IndexType, memref_type, RR: int = 8, elem_bytes: int = 4, suffix: str = "fp32"
) -> ModuleOp:
    """Build MLIR module with both pack_CB_v branches."""
    return ModuleOp(
        [
            build_pack_CB_v_if(idx, memref_type, RR, elem_bytes, suffix),
            build_pack_CB_v_else(idx, memref_type, RR, elem_bytes, suffix),
        ]
    )


def build_pack_module(
    idx: IndexType,
    memref_type,
    NR: int = 8,
    MR: int = 4,
    elem_bytes: int = 4,
    suffix: str = "fp32",
) -> ModuleOp:
    """Build MLIR module with ALL packing branches (CB and RB)."""
    return ModuleOp(
        [
            build_pack_CB_v_if(idx, memref_type, NR, elem_bytes, suffix),
            build_pack_CB_v_else(idx, memref_type, NR, elem_bytes, suffix),
            build_pack_RB_v_if(idx, memref_type, MR, elem_bytes, suffix),
            build_pack_RB_v_else(idx, memref_type, MR, elem_bytes, suffix),
        ]
    )
