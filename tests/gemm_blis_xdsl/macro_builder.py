"""
macro_builder.py

Builder for the GEMM Macro-Kernel loops (BLIS B3A2C0 flavor).
Generates the 5-nested loop structure in xDSL.
Supports compile-time specialization for matrix layouts.
"""

from xdsl.dialects import arith, builtin, emitc, func, memref, scf
from xdsl.dialects.builtin import (Float32Type, IndexType, MemRefType, i32, i64, 
                                   ModuleOp, FunctionType, FloatAttr, IntegerAttr)
from xdsl.ir import Block, Region, SSAValue

class MacroBuilder:
    def __init__(self, block: Block):
        self.block = block
    
    def const_i32(self, val: int) -> SSAValue:
        op = arith.ConstantOp(IntegerAttr(val, i32))
        self.block.add_op(op)
        return op.results[0]

    def const_f32(self, val: float) -> SSAValue:
        op = arith.ConstantOp(FloatAttr(val, Float32Type()))
        self.block.add_op(op)
        return op.results[0]

    def sub(self, lhs: SSAValue, rhs: SSAValue) -> SSAValue:
        op = arith.SubiOp(lhs, rhs)
        self.block.add_op(op)
        return op.results[0]

    def add(self, lhs: SSAValue, rhs: SSAValue) -> SSAValue:
        op = arith.AddiOp(lhs, rhs)
        self.block.add_op(op)
        return op.results[0]

    def mul(self, lhs: SSAValue, rhs: SSAValue) -> SSAValue:
        op = arith.MuliOp(lhs, rhs)
        self.block.add_op(op)
        return op.results[0]

    def min(self, val1: SSAValue, val2_int: int) -> SSAValue:
        c_val2 = self.const_i32(val2_int)
        op = arith.MinSIOp(val1, c_val2)
        self.block.add_op(op)
        return op.results[0]

    def cmpi(self, lhs: SSAValue, rhs: SSAValue, predicate: int) -> SSAValue:
        op = arith.CmpiOp(lhs, rhs, predicate)
        self.block.add_op(op)
        return op.results[0]

    def get_ptr(self, memref_val: SSAValue, offset: SSAValue) -> SSAValue:
        ptr_t = emitc.EmitC_PointerType(Float32Type())
        op = emitc.EmitC_CallOpaqueOp("GET_PTR", [memref_val, offset], 
                                       result_types=[ptr_t])
        self.block.add_op(op)
        return op.results[0]

    def call_opaque(self, name: str, args: list):
        op = emitc.EmitC_CallOpaqueOp(name, args, [])
        self.block.add_op(op)

    def call_opaque_with_result(self, name: str, args: list, ret_type) -> SSAValue:
        op = emitc.EmitC_CallOpaqueOp(name, args, [ret_type])
        self.block.add_op(op)
        return op.results[0]

    def calc_offset(self, row: SSAValue, col: SSAValue, ld: SSAValue, layout: str) -> SSAValue:
        if layout == 'C':
            return self.add(self.mul(col, ld), row)
        else:
            return self.add(self.mul(row, ld), col)

def build_macro_module(MC: int = 256, NC: int = 1024, KC: int = 256,
                       MR: int = 4,   NR: int = 8,
                       orderA: str = 'C', transA: str = 'N',
                       orderB: str = 'C', transB: str = 'N',
                       orderC: str = 'C',
                       suffix: str = "fp32") -> ModuleOp:
    
    f32 = Float32Type()
    mem_f32 = MemRefType(f32, [-1])
    
    # Function Signature
    fn_name = f"gemm_blis_v_{orderA}{transA}_{orderB}{transB}_{orderC}_{suffix}"
    arg_types = [i32, i32, i32, f32, mem_f32, i32, mem_f32, i32, f32, mem_f32, i32, mem_f32, mem_f32]
    
    # Create the function with an explicit Region and Block
    entry_block = Block(arg_types=arg_types)
    macro_fn = func.FuncOp(fn_name, (arg_types, []), Region([entry_block]))
    
    m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, Ac, Bc = entry_block.args
    
    mb = MacroBuilder(entry_block)
    c0 = mb.const_i32(0)
    
    # ── JC Loop (outermost): step NC, iterates 0..n → jc = N-panel offset ──────
    # ForOp: entry_block.ForOp(0, n, NC) → body = jc_body, induction var = jc
    nc_step = mb.const_i32(NC)
    jc_body = Block(arg_types=[i32])
    jc = jc_body.args[0]          # jc: N-panel start index (0, NC, 2*NC, ...)
    jc_builder = MacroBuilder(jc_body)
    nc = jc_builder.min(jc_builder.sub(n, jc), NC)  # nc = min(n-jc, NC)
    
    # ── PC Loop (2nd): step KC, iterates 0..k → pc = K-panel offset ──────────
    # ForOp: jc_body.ForOp(0, k, KC) → body = pc_body, induction var = pc
    kc_step = jc_builder.const_i32(KC)
    pc_body = Block(arg_types=[i32])
    pc = pc_body.args[0]          # pc: K-panel start index (0, KC, 2*KC, ...)
    pc_builder = MacroBuilder(pc_body)
    kc = pc_builder.min(pc_builder.sub(k, pc), KC)  # kc = min(k-pc, KC)
    
    # Pointer Math for B
    row_B, col_B = (pc, jc) if transB == 'N' else (jc, pc)
    off_B = pc_builder.calc_offset(row_B, col_B, ldB, orderB)
    Bptr = pc_builder.get_ptr(B, off_B)
    
    # Branchless vectorized packing function name
    pack_CB_name = f"pack_CB_v_{'if' if orderB == 'C' and transB == 'N' else 'else'}_{suffix}_RR{NR}"
    pc_builder.call_opaque(pack_CB_name, [Bptr, Bc, kc, nc, ldB])
    
    # Beta handling
    betaI = pc_builder.call_opaque_with_result("GET_BETA", [pc, beta], f32)
    
    # ── IC Loop (3rd): step MC, iterates 0..m → ic = M-panel offset ──────────
    # ForOp: pc_body.ForOp(0, m, MC) → body = ic_body, induction var = ic
    mc_step = pc_builder.const_i32(MC)
    ic_body = Block(arg_types=[i32])
    ic = ic_body.args[0]          # ic: M-panel start index (0, MC, 2*MC, ...)
    ic_builder = MacroBuilder(ic_body)
    mc = ic_builder.min(ic_builder.sub(m, ic), MC)  # mc = min(m-ic, MC)
    
    row_A, col_A = (ic, pc) if transA == 'N' else (pc, ic)
    off_A = ic_builder.calc_offset(row_A, col_A, ldA, orderA)
    Aptr = ic_builder.get_ptr(A, off_A)
    
    # Branchless vectorized packing function name
    pack_RB_name = f"pack_RB_v_{'if' if orderA == 'C' and transA == 'N' else 'else'}_{suffix}_RR{MR}"
    ic_builder.call_opaque(pack_RB_name, [Aptr, Ac, mc, kc, ldA])
    
    # ── JR Loop (4th): step NR, iterates 0..nc → jr = micro-N offset ─────────
    # ForOp: ic_body.ForOp(0, nc, NR) → body = jr_body, induction var = jr
    nr_step = ic_builder.const_i32(NR)
    jr_body = Block(arg_types=[i32])
    jr = jr_body.args[0]          # jr: micro-tile N offset (0, NR, 2*NR, ...)
    jr_builder = MacroBuilder(jr_body)
    nr = jr_builder.min(jr_builder.sub(nc, jr), NR)  # nr = min(nc-jr, NR)
    
    # ── IR Loop (innermost): step MR, iterates 0..mc → ir = micro-M offset ────
    # ForOp: jr_body.ForOp(0, mc, MR) → body = ir_body, induction var = ir
    mr_step = jr_builder.const_i32(MR)
    ir_body = Block(arg_types=[i32])
    ir = ir_body.args[0]          # ir: micro-tile M offset (0, MR, 2*MR, ...)
    ir_builder = MacroBuilder(ir_body)
    mr = ir_builder.min(ir_builder.sub(mc, ir), MR)  # mr = min(mc-ir, MR)
    
    # Pointer Math for C
    row_C = ir_builder.add(ic, ir)
    col_C = ir_builder.add(jc, jr)
    off_C = ir_builder.calc_offset(row_C, col_C, ldC, orderC)
    Cptr = ir_builder.get_ptr(C, off_C)
    
    # Pack buffers sub-pointers
    off_Ac = ir_builder.mul(ir, kc)
    Ac_ptr = ir_builder.get_ptr(Ac, off_Ac)
    off_Bc = ir_builder.mul(jr, kc)
    Bc_ptr = ir_builder.get_ptr(Bc, off_Bc)
    
    ir_builder.call_opaque("gemm_microkernel_v", 
                           [mr, nr, kc, alpha, Ac_ptr, Bc_ptr, betaI, Cptr, ldC])
    
    # ── Wire loop nest (inside-out): each body block receives its ForOp ────────
    # IR (innermost): jr_body.ForOp(0, mc, MR) → iterates ir over mc
    ir_body.add_op(scf.YieldOp())
    jr_body.add_op(scf.ForOp(c0, mc, mr_step, [], Region([ir_body])))
    # JR: ic_body.ForOp(0, nc, NR) → iterates jr over nc
    jr_body.add_op(scf.YieldOp())
    ic_body.add_op(scf.ForOp(c0, nc, nr_step, [], Region([jr_body])))
    # IC: pc_body.ForOp(0, m, MC) → iterates ic over m
    ic_body.add_op(scf.YieldOp())
    pc_body.add_op(scf.ForOp(c0, m, mc_step, [], Region([ic_body])))
    # PC: jc_body.ForOp(0, k, KC) → iterates pc over k
    pc_body.add_op(scf.YieldOp())
    jc_body.add_op(scf.ForOp(c0, k, kc_step, [], Region([pc_body])))
    # JC (outermost): entry.ForOp(0, n, NC) → iterates jc over n
    jc_body.add_op(scf.YieldOp())
    entry_block.add_op(scf.ForOp(c0, n, nc_step, [], Region([jc_body])))
    
    entry_block.add_op(func.ReturnOp())
    
    return ModuleOp([macro_fn])
