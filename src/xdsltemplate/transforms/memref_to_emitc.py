# src/xdsltemplate/transforms/memref_to_emitc.py

from xdsl.context import Context
from xdsl.dialects import emitc, func, memref
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ConvertMemRefTypeToEmitCPtr(RewritePattern):
    """Convert memref types to emitc.ptr in function signatures"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        # Check if any argument is a memref
        has_memref = any(
            isinstance(arg.type, memref.MemRefType) for arg in op.body.block.args
        )

        if not has_memref:
            return

        # Convert argument types
        new_arg_types = []
        for arg in op.body.block.args:
            if isinstance(arg.type, memref.MemRefType):
                elem_type = arg.type.element_type
                # EmitC_PointerType takes a single type parameter
                new_type = emitc.EmitC_PointerType(elem_type)
                new_arg_types.append(new_type)
            else:
                new_arg_types.append(arg.type)

        # Create new function type
        new_func_type = func.FunctionType.from_lists(
            new_arg_types,
            # list(op.function_type.outputs)
            list(op.function_type.outputs.data),
        )

        # Create new block with updated argument types
        new_block = Block(arg_types=new_arg_types)

        # Map old block arguments to new ones
        for old_arg, new_arg in zip(op.body.block.args, new_block.args):
            old_arg.replace_by(new_arg)

        # Move operations from old block to new block
        for op_inner in list(op.body.block.ops):
            op_inner.detach()
            new_block.add_op(op_inner)

        # Create new region with the new block
        new_region = Region()
        new_region.add_block(new_block)

        # Create new function operation
        new_func = func.FuncOp(
            name=op.sym_name.data,
            function_type=new_func_type,
            region=new_region,
        )

        # --- NEW SECTION: return the function type ---
        inputs = list(new_func_type.inputs)
        outputs = list(new_func_type.outputs)
        if len(inputs) >= 3 and len(outputs) == 0:
            # obtain the third argument type
            arg2_type = inputs[2]

            if hasattr(arg2_type, "element_type"):
                return_type = arg2_type.element_type
            elif hasattr(arg2_type, "pointee_type"):
                return_type = arg2_type.pointee_type
            else:
                return_type = arg2_type

            # Build new type with return value
            # fixed_func_type = func.FunctionType.from_lists(inputs, [return_type])
            # new_func.function_type = fixed_func_type

            fixed_func_type = func.FunctionType.from_lists(inputs, [return_type])
            new_func.attributes["function_type"] = fixed_func_type

        # Copy other attributes
        for attr_name, attr_value in op.attributes.items():
            if attr_name not in ("sym_name", "function_type"):
                new_func.attributes[attr_name] = attr_value

        # Replace the old function with the new one
        # func.FuncOp has no results, so we pass empty list
        rewriter.replace_op(op, new_func, new_results=[])


class MemRefToEmitCPass(ModulePass):
    """Pass to convert memref operations to EmitC"""

    name = "memref-to-emitc"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        # First, apply the memref → emitc.ptr rewrite
        PatternRewriteWalker(
            ConvertMemRefTypeToEmitCPtr(), apply_recursively=True
        ).rewrite_module(op)
