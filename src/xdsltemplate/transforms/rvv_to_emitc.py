from xdsl.context import Context
from xdsl.dialects import emitc
from xdsl.dialects.builtin import Float32Type, IntegerType, ModuleOp, StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from xdsltemplate.dialects.emitc_ext import EmitCSubscriptOp
from xdsltemplate.dialects.rvv import *

# from xdsltemplate.dialects.rvv import *


class ConvertRVVSetvlToEmitC(RewritePattern):
    """
    %vl = "rvv.setvl"(%avl) → __riscv_vsetvl_e32m1(avl)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SetvlOp, rewriter: PatternRewriter):
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vsetvl_e32m1",
            call_args=[op.avl],
            result_types=[op.vl.type],
        )
        rewriter.replace_op(op, [call_op])


class ConvertRVVSetvlMf4ToEmitC(RewritePattern):
    """
    %vl = "rvv.setvl_mf4"(%avl) → __riscv_vsetvl_e8mf4(avl)
    Correct setvl for vint8mf4_t / vint16mf2_t / vint32m1_t mixed precision.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SetvlMf4Op, rewriter: PatternRewriter):
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vsetvl_e8mf4",
            call_args=[op.avl],
            result_types=[op.vl.type],
        )
        rewriter.replace_op(op, [call_op])


class ConvertRVV_vle32_v_f32m1_ToEmitC(RewritePattern):
    """
    Lower
    We get
          %va = "rvv.vle32_v_f32m1Op"(%arg0, %vl) : (memref<?xf32>, index) -> f32
      secondery
          %va = "rvv.vle32_v_f32m1Op"(%arg0, %vl) : (!emitc.ptr<f32>, index) -> f32
        to
        %va = emitc.call_opaque "__riscv_vle32_v_f32m1"(%arg0, %vl)
              : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
               Then need to be lowered emitc call op
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vle32_v_f32m1Op, rewriter: PatternRewriter):
        lvalue_type = emitc.EmitC_LValueType(Float32Type())
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)

        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[op.memref.type],
            attributes={"applicableOperator": StringAttr("&")},
        )
        # // Custom form of applying the & operator.
        # %0 = emitc.apply "&"(%arg0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
        #
        # // Generic form of the same operation.
        # %0 = "emitc.apply"(%arg0) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

        # Create RVV call with the address
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vfloat32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vle32_v_f32m1",
            call_args=[ptr_ref.result, op.avl],
            # result_types=[op.vl.type],
            result_types=[vector_type],
        )
        # Replace with all three operations
        rewriter.replace_op(op, [subscript, ptr_ref, call_op])


class ConvertRVV_vlse32_v_f32m1_ToEmitC(RewritePattern):
    """
    Lower
        %va = "rvv.vlse32_v_f32m1Op"(%arg0, %offset, %stride, %vl)
    to
        %va = emitc.call_opaque "__riscv_vlse32_v_f32m1"(&arg0[offset], stride, vl)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vlse32_v_f32m1Op, rewriter: PatternRewriter):
        lvalue_type = emitc.EmitC_LValueType(Float32Type())
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)

        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[op.memref.type],
            attributes={"applicableOperator": StringAttr("&")},
        )

        vector_type = emitc.EmitC_OpaqueType(StringAttr("vfloat32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vlse32_v_f32m1",
            call_args=[ptr_ref.result, op.bstride, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, [subscript, ptr_ref, call_op])


from xdsltemplate.dialects.emitc_ext import EmitCLoadOp


class ConvertRVV_vfmacc_vf_f32m1_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vfmacc_vf_f32m1Op, rewriter: PatternRewriter):
        # Create subscript to get lvalue<f32>
        lvalue_type = emitc.EmitC_LValueType(Float32Type())
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)
        rewriter.insert_op_before_matched_op(subscript)

        # Load the lvalue to get f32 value
        load_op = EmitCLoadOp(src=subscript.result, result_type=Float32Type())
        rewriter.insert_op_before_matched_op(load_op)

        # Now use the loaded f32 value in the call
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vfloat32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vfmacc_vf_f32m1",
            call_args=[
                op.vd,
                load_op.result,
                op.vs,
                op.avl,
            ],  # Use load_op.result instead of subscript.result
            result_types=[vector_type],
        )

        rewriter.replace_op(op, call_op)


class ConvertRVV_vfmv_v_f_f32m1_ToEmitC(RewritePattern):
    """
        Convert:
        %v = "rvv.vfmv_v_f_f32m1"(%scalar, %vl) : (f32, index) -> !rvv.vec<f32, 1>
    To:
        %v = emitc.call_opaque "__riscv_vfmv_v_f_f32m1"(%scalar, %vl)
            : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vfmv_v_f_f32m1Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vfloat32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vfmv_v_f_f32m1",
            call_args=[op.scalar, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)

        return


class ConvertRVV_vse32_v_f32m1_ToEmitC(RewritePattern):
    """
    Convert
        rvv.vse32_v_f32m1( %memeref, %offset, %vec, %vl) : (memref<?xf32>, index, !rvv.vfloat32m1, index) -> ()
    To
     __riscv_vse32_v_f32m1(&C[0], C_reg_0_0, (vl));

     To find the pointer via emitc-subscript function, get the lvalue of the memeref location acording to the offset
     Then use app function to return a emitc ptr
     then use emitc call function to call a c func with ptr address, vecter name and vl
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vse32_v_f32m1Op, rewriter: PatternRewriter):
        # Get address: &C[offset]
        lvalue_type = emitc.EmitC_LValueType(Float32Type())
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)
        rewriter.insert_op_before_matched_op(subscript)

        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[op.memref.type],
            attributes={"applicableOperator": StringAttr("&")},
        )
        rewriter.insert_op_before_matched_op(ptr_ref)

        # Call the store intrinsic (no result)
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vse32_v_f32m1",
            call_args=[ptr_ref.result, op.vec, op.avl],
            result_types=[],  # No result for store
        )

        rewriter.replace_op(op, call_op)


#     C00 = __riscv_vmv_v_x_i32m4(0, vlength);
class ConvertRVV_vmv_v_x_i32m4_ToEmitC(RewritePattern):
    """
    We want to convert
        rvv.vmv_v_x_i32m4Op( %scarler, %vl): (INT32, Index) -> !rvv.RVVInt32M4Type
    To
        %C00 = emitc.call_opaque "__riscv_vmv_v_x_i32m4"(%scalar, %vl)
            : (int32, index) -> !emitc.opaque<"vint32m4_t">
    To
        vint32m4_t C00 = __riscv_vmv_v_x_i32m4(0, vlength);
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vmv_v_x_i32m4Op, rewriter: PatternRewriter):
        # extract operands
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint32m4_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vmv_v_x_i32m4",
            call_args=[op.scalar, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


#     vtmp = __riscv_vle8_v_i8m1(&Ar[bA + 0], vlength);
class ConvertRVV_vle8_v_i8m1_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vle8_v_i8m1Op, rewriter: PatternRewriter):
        lvalue_type = emitc.EmitC_LValueType(IntegerType(8))
        # this is a potentional hazard, int8 type does not predefined
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)
        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[op.memref.type],
            attributes={"applicableOperator": StringAttr("&")},
        )
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint8m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vle8_v_i8m1",
            call_args=[ptr_ref.result, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, [subscript, ptr_ref, call_op])


class ConvertRVV_vle32_v_i32m4_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vle32_v_i32m4Op, rewriter: PatternRewriter):
        lvalue_type = emitc.EmitC_LValueType(IntegerType(32))
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)
        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[op.memref.type],
            attributes={"applicableOperator": StringAttr("&")},
        )
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint32m4_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vle32_v_i32m4",
            call_args=[ptr_ref.result, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, [subscript, ptr_ref, call_op])


class ConvertRVV_vwadd_vx_i16m2_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vwadd_vx_i16m2Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint16m2_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vwadd_vx_i16m2",
            call_args=[op.vs, op.scalar, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vwmacc_vx_i32m4_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vwmacc_vx_i32m4Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint32m4_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vwmacc_vx_i32m4",
            call_args=[op.vd, op.scalar, op.vs2, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vse32_v_i32m4_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vse32_v_i32m4Op, rewriter: PatternRewriter):
        lvalue_type = emitc.EmitC_LValueType(IntegerType(32))
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)
        rewriter.insert_op_before_matched_op(subscript)
        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[op.memref.type],
            attributes={"applicableOperator": StringAttr("&")},
        )
        rewriter.insert_op_before_matched_op(ptr_ref)
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vse32_v_i32m4",
            call_args=[ptr_ref.result, op.vec, op.avl],
            result_types=[],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vmv_v_x_i32m1_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vmv_v_x_i32m1Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vmv_v_x_i32m1",
            call_args=[op.scalar, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vle8_v_i8mf4_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vle8_v_i8mf4Op, rewriter: PatternRewriter):
        lvalue_type = emitc.EmitC_LValueType(IntegerType(8))
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)
        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[op.memref.type],
            attributes={"applicableOperator": StringAttr("&")},
        )
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint8mf4_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vle8_v_i8mf4",
            call_args=[ptr_ref.result, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, [subscript, ptr_ref, call_op])


class ConvertRVV_vwmul_vv_i16mf2_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vwmul_vv_i16mf2Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint16mf2_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vwmul_vv_i16mf2",
            call_args=[op.vs2, op.vs1, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vmv_v_x_i16mf2_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vmv_v_x_i16mf2Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint16mf2_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vmv_v_x_i16mf2",
            call_args=[op.scalar, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vwmacc_vv_i16mf2_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vwmacc_vv_i16mf2Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint16mf2_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vwmacc_vv_i16mf2",
            call_args=[op.vd, op.vs1, op.vs2, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vwmacc_vx_i16mf2_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vwmacc_vx_i16mf2Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint16mf2_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vwmacc_vx_i16mf2",
            call_args=[op.vd, op.rs1, op.vs2, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vsext_vf2_i16mf2_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vsext_vf2_i16mf2Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint16mf2_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vsext_vf2_i16mf2",
            call_args=[op.vs2, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vwmacc_vx_i32m1_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vwmacc_vx_i32m1Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vwmacc_vx_i32m1",
            call_args=[op.vd, op.rs1, op.vs2, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vwadd_wv_i32m1_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vwadd_wv_i32m1Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vwadd_wv_i32m1",
            call_args=[op.vs2, op.vs1, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)


class ConvertRVV_vle32_v_i32m1_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vle32_v_i32m1Op, rewriter: PatternRewriter):
        lvalue_type = emitc.EmitC_LValueType(IntegerType(32))
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)
        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[op.memref.type],
            attributes={"applicableOperator": StringAttr("&")},
        )
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vle32_v_i32m1",
            call_args=[ptr_ref.result, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, [subscript, ptr_ref, call_op])


class ConvertRVV_vse32_v_i32m1_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vse32_v_i32m1Op, rewriter: PatternRewriter):
        lvalue_type = emitc.EmitC_LValueType(IntegerType(32))
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)
        rewriter.insert_op_before_matched_op(subscript)
        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[op.memref.type],
            attributes={"applicableOperator": StringAttr("&")},
        )
        rewriter.insert_op_before_matched_op(ptr_ref)
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vse32_v_i32m1",
            call_args=[ptr_ref.result, op.vec, op.avl],
            result_types=[],
        )
        rewriter.replace_op(op, call_op)


class RVVToEmitCPass(ModulePass):
    name = "rvv_to_emitc"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        patterns = [
            ConvertRVVSetvlToEmitC(),
            ConvertRVVSetvlMf4ToEmitC(),
            ConvertRVV_vle32_v_f32m1_ToEmitC(),
            ConvertRVV_vlse32_v_f32m1_ToEmitC(),
            ConvertRVV_vfmacc_vf_f32m1_ToEmitC(),
            ConvertRVV_vfmv_v_f_f32m1_ToEmitC(),
            ConvertRVV_vse32_v_f32m1_ToEmitC(),
            # Int
            ConvertRVV_vmv_v_x_i32m4_ToEmitC(),
            ConvertRVV_vle8_v_i8m1_ToEmitC(),
            ConvertRVV_vle32_v_i32m4_ToEmitC(),
            ConvertRVV_vwadd_vx_i16m2_ToEmitC(),
            ConvertRVV_vwmacc_vx_i32m4_ToEmitC(),
            ConvertRVV_vse32_v_i32m4_ToEmitC(),
            # Mixed Precision
            ConvertRVV_vle8_v_i8mf4_ToEmitC(),
            ConvertRVV_vwmul_vv_i16mf2_ToEmitC(),
            ConvertRVV_vmv_v_x_i16mf2_ToEmitC(),
            ConvertRVV_vwmacc_vv_i16mf2_ToEmitC(),
            ConvertRVV_vwmacc_vx_i16mf2_ToEmitC(),
            ConvertRVV_vsext_vf2_i16mf2_ToEmitC(),
            ConvertRVV_vwmacc_vx_i32m1_ToEmitC(),
            ConvertRVV_vwadd_wv_i32m1_ToEmitC(),
            ConvertRVV_vle32_v_i32m1_ToEmitC(),
            ConvertRVV_vse32_v_i32m1_ToEmitC(),
            ConvertRVV_vmv_v_x_i32m1_ToEmitC(),
        ]
        for pattern in patterns:
            PatternRewriteWalker(pattern, apply_recursively=True).rewrite_module(op)
