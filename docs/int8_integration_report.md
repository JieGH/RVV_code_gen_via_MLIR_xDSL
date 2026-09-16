# INT8 RVV Dialect Integration and Verification Report

This report documents the extensions added to the `xdsltemplate` compiler to support INT8 tensor operations on RISC-V.

## 1. Developer Workflow Overview (Tutorial)
If you are aiming to recreate this design or add your own custom hardware instructions, follow this step-by-step pipeline framework:

1. **Define the Types (File: `rvv.py`)**: Inherit from `ParametrizedAttribute` to define the vector registers (e.g. `RVVInt8M1Type`) the hardware interacts with.
2. **Define the Operations (File: `rvv.py`)**: Inherit from `IRDLOperation` and strictly define the `.operands` and `.result` signatures. 
3. **Register the Dialect (File: `rvv.py`)**: At the bottom of the file, instantiate a `Dialect` object listing every new custom type and operation so the `Context` recognizes them.
4. **Implement Lowering Rules (File: `rvv_to_emitc.py`)**: Write a class inheriting from `RewritePattern` for each new intrinsic. It must `replace_op` your high-level operation with `emitc.call_opaque` matching exactly the signature of the target C header intrinsic.
5. **Register the Passes (File: `rvv_to_emitc.py`)**: Append each class definition to the `patterns=[]` list inside `RVVToEmitCPass`.
6. **Integrate into Builder (Python API)**: Call the operation wrappers defined in step 2 procedurally to build standard IR blocks.
7. **Verify & Translate**: Run translation testing (`mlir-translate -mlir-to-cpp`) ensuring all high-level `arith` and `memref` primitives are fully lowered so the downstream EmitC C++ emitter recognizes them.

---

## 2. Dialect Types and Operations Added

To represent the hardware precisely, we extended the definitions in `src/xdsltemplate/dialects/rvv.py`. The `xDSL` framework requires all new types to be explicitly registered classes.

### How New Register Types Were Defined
We introduced new data types representing `vint8m1_t`, `vint16m2_t`, and `vint32m4_t` register groups. Because these are custom primitives, we define them using xDSL's `ParametrizedAttribute` and `TypeAttribute` inheritance framework. 

**Example (Lines 54-58 in `rvv.py`):**
```python
@irdl_attr_definition
class RVVInt16M2Type(ParametrizedAttribute, TypeAttribute):
    name = "rvv.vint16m2"
```

### How New Intrinsic Operations Were Mapped
Each hardware instruction is defined as a uniquely typed `IRDLOperation`. Using the `@irdl_op_definition` decorator, we mandate what input types the operation naturally binds to and what it outputs.

#### INT8 Load Operation (`vle8_v_i8m1Op`)
**File:** `src/xdsltemplate/dialects/rvv.py` (Lines 91-106)
This explicitly defines that loading a memref returns the `RVVInt8M1Type`.
```python
@irdl_op_definition
class vle8_v_i8m1Op(IRDLOperation):
    name = "rvv.vle8_v_i8m1Op"
    memref = operand_def(MemRefType)
    offset = operand_def(IndexType)
    avl = operand_def(IndexType)
    result = result_def(RVVInt8M1Type)
    
    def __init__(self, memref: SSAValue, offset: SSAValue, avl: SSAValue):
        super().__init__(
            operands=[memref, offset, avl],
            result_types=[RVVInt8M1Type()]
        )
```

---

## 2. Using the API to Build IR Functions

Now that the types and operations are defined, we can invoke their classes entirely through python to procedurally construct our RISC-V compute graphs. 

Here is an example python script using the newly defined `rvv.py` operations to construct a widening multiplication block (like building `tests/verify_int8_rvv.py` or the `build_gemm_int8` pipeline):

```python
# 1. First, instantiate your loop variables and pointer states
Ar_ptr = entry_block.args[0]
Br_ptr = entry_block.args[1]
vl_val = arith.ConstantOp(IntegerAttr(32, IndexType())).result
i0     = arith.ConstantOp(IntegerAttr(0, IndexType())).result
i8_0   = arith.ConstantOp(IntegerAttr(0, IntegerType(8))).result

# 2. Invoke the operations directly as python objects! 
# They consume variables (SSAValues) and spit out new ones (.result)

# Load A vector (returns vint8m1_t)
vle8_op = vle8_v_i8m1Op(Ar_ptr, i0, vl_val)
entry_block.add_op(vle8_op)

# Widen A vector to i16 (returns vint16m2_t)
vwadd_op = vwadd_vx_i16m2Op(vle8_op.result, i8_0, vl_val)
entry_block.add_op(vwadd_op)

# Issue the core Widening MAC
vwmacc_op = vwmacc_vx_i32m4Op(
    C00_acc,                # Prior i32 Accumulator
    memref.load(Br, i0),    # i8 Scalar
    vwadd_op.result,        # The widened A vector
    vl_val                  # Vector length
)
entry_block.add_op(vwmacc_op)
```

### INT8 Widening Multiply-Accumulate (`vwmacc_vx_i32m4Op`)
**File:** `src/xdsltemplate/dialects/rvv.py` (Lines 156-168)
The core compute instruction that calculates `C += A * B`. It takes the widened `i16` vector, an `i8` scalar, and updates the `i32` accumulator. 
```python
@irdl_op_definition
class vwmacc_vx_i32m4Op(IRDLOperation):
    """__riscv_vwmacc_vx_i32m4(vd, scalar_i8, vs2_i16, vl)
        C00 = __riscv_vwmacc_vx_i32m4(C00, Br[bB + 0], A0, vlength);
    """
    name = "rvv.vwmacc_vx_i32m4"
    vd     = operand_def(RVVInt32M4Type)
    scalar = operand_def(IntegerType)
    vs2    = operand_def(RVVInt16M2Type)
    avl    = operand_def(IndexType)
    result = result_def(RVVInt32M4Type)
    ...
```

---

## 2. EmitC Lowering Patterns

To emit valid C code, the abstract mathematical operations are converted into `emitc.call_opaque` function calls.

### Array Subscript Lowering (`vle8`)
**File:** `src/xdsltemplate/transforms/rvv_to_emitc.py` (Lines 197-215)
To extract a C-pointer `&Ar[offset]` from the `memref`, we decompose the operation into a subscript, an address-of (`&`) operator, and finally the intrinsic call.
```python
class ConvertRVV_vle8_v_i8m1_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vle8_v_i8m1Op, rewriter: PatternRewriter):
        from xdsl.dialects.builtin import IntegerType
        
        # 1. Subscript: Ar[offset]
        lvalue_type = emitc.EmitC_LValueType(IntegerType(8))
        subscript = EmitCSubscriptOp(op.memref, op.offset, lvalue_type)
        
        # 2. Apply '&' address-of operator
        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[op.memref.type],
            attributes={"applicableOperator": StringAttr("&")},
        )
        
        # 3. Call the intrinsic
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vint8m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vle8_v_i8m1",
            call_args=[ptr_ref.result, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, [subscript, ptr_ref, call_op])
```

### Scalar Compute Lowering (`vwmacc`)
**File:** `src/xdsltemplate/transforms/rvv_to_emitc.py` (Lines 247-256)
Because `vwmacc` deals strictly with scalar integers and initialized vectors rather than memory references, the translation is a straightforward 1-to-1 map.
```python
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
```

---

## 3. Translation Output and Fixes

During verification in `tests/verify_int8_rvv.py`, the `mlir-translate` executable rejected the converted output. 

Two critical issues were resolved to fix compilation:
1. **Pass completeness**: Standard MLIR constructs like `arith.constant` and `memref.load` must be lowered using `ArithToEmitCPass` and `MemrefLoadToEmitcPass` respectively before translation. If they are left un-lowered, the `emitc` compiler drops them.
2. **Generic Format Parseing**: xDSL inherently generates "pretty-printed" MLIR code. `mlir-translate` from the standard un-registered LLVM suite expects highly explicit generic MLIR blocks. Setting `print_generic_format=True` on the `xDSL Printer` resolves this.

### Final Translated C++ Implementation

After successful compilation from xDSL to MLIR to EmitC to CPP, the final result looks like this:
```c
#include <stdint.h>

void test_int8_ops(int8_t* v1, int8_t* v2, int32_t* v3) {
  size_t v4 = 32;
  size_t v5 = 0;
  int8_t v6 = 0;
  int32_t v7 = 0;
  
  // Initialize Accumulator
  vint32m4_t v8 = __riscv_vmv_v_x_i32m4(v7, v4);
  int32_t* v9 = &v3[v5];
  vint32m4_t v10 = __riscv_vle32_v_i32m4(v9, v4);
  
  // Load Matrix A
  int8_t* v11 = &v1[v5];
  vint8m1_t v12 = __riscv_vle8_v_i8m1(v11, v4);
  
  // Widen to i16
  vint16m2_t v13 = __riscv_vwadd_vx_i16m2(v12, v6, v4);
  
  // Load Matrix B Scalar
  int8_t v14 = v2[v5];
  
  // Widen Mac Compute (A * B + C)
  vint32m4_t v15 = __riscv_vwmacc_vx_i32m4(v8, v14, v13, v4);
  
  // Final Store Output
  int32_t* v16 = &v3[v5];
  __riscv_vse32_v_i32m4(v16, v15, v4);
  
  return;
}
```
