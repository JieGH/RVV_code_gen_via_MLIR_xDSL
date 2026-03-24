from xdsl.dialects.builtin import (
    Float32Type,
    IndexType,
    IntegerType,
    MemRefType,
)
from xdsl.ir import Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)


# Since we need tp define rvv type
# like   vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
@irdl_attr_definition
class RVVvectorType(ParametrizedAttribute):
    # class RVVVectorType(ParametrizedAttribute, TypeAttribute):
    """RVV custom type - represents vfloat32m1_t"""
    name = "rvv.vec"


@irdl_attr_definition
# class RVVFloat32M1Type(ParametrizedAttribute):
# class RVVFloat32M1Type(ParametrizedAttribute):
class RVVFloat32M1Type(ParametrizedAttribute, TypeAttribute):
    name = "rvv.vfloat32m1"


@irdl_attr_definition
class RVVFloat32M2Type(ParametrizedAttribute,TypeAttribute):
    """Represents vfloat32m2_t"""
    name = "rvv.vfloat32m2"

@irdl_attr_definition
class RVVInt8M1Type(ParametrizedAttribute, TypeAttribute):
    """Represents vint8m1_t"""
    name = "rvv.vint8m1"

@irdl_attr_definition
class RVVInt16M2Type(ParametrizedAttribute, TypeAttribute):
    """Represents vint16m2_t"""
    name = "rvv.vint16m2"

@irdl_attr_definition
class RVVInt32M4Type(ParametrizedAttribute, TypeAttribute):
    """Represents vint32m4_t"""
    name = "rvv.vint32m4"


@irdl_op_definition
class SetvlOp(IRDLOperation):
    """
    Set vector length
    """

    name = "rvv.setvl"

    # Application vector length, we pass on the size of the index
    avl = operand_def(IndexType)
    vl = result_def(IndexType)

    def __init__(self, avl: SSAValue):
        super().__init__(
            operands=[avl],
            result_types=[IndexType()],
        )


@irdl_op_definition
class vle32_v_f32m1Op(IRDLOperation):
    """Vector load operation"""
    name = "rvv.vle32_v_f32m1Op"
    memref = operand_def(MemRefType)
    offset = operand_def(IndexType)
    avl = operand_def(IndexType)
    # vl = result_def(RVVvectorType)
    vl = result_def(RVVFloat32M1Type)

    def __init__(self, memref: SSAValue, offset: SSAValue, avl: SSAValue):
        super().__init__(
            operands=[memref, offset, avl],
            # result_types=[RVVFloat32M1Type([])],  # No parameters!
            result_types=[RVVFloat32M1Type()],  # No parameters!
            # result_types=[RVVvectorType([Float32Type(), IntAttr(1)])],  # Use list constructor
        )

@irdl_op_definition
class vle8_v_i8m1Op(IRDLOperation):
    """
    Vector load operation for vint8m1_t
    vtmp = __riscv_vle8_v_i8m1(&Ar[bA + 0], vlength);
    """
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

@irdl_op_definition
class vle32_v_i32m4Op(IRDLOperation):
    """
    Vector load operation from vint32m4_t
    C00 = __riscv_vle32_v_i32m4(&Ccol(0, 0), vlength);
    """
    name = "rvv.vle32_v_i32m4Op"
    memref = operand_def(MemRefType)
    offset = operand_def(IndexType)
    avl = operand_def(IndexType)
    result = result_def(RVVInt32M4Type)
    def __init__(self, memref: SSAValue, offset: SSAValue, avl: SSAValue):
        super().__init__(
            operands=[memref, offset, avl],
            result_types=[RVVInt32M4Type()]
        )

@irdl_op_definition
class vwadd_vx_i16m2Op(IRDLOperation):
    """
    A0   = __riscv_vwadd_vx_i16m2(vtmp, 0,  vlength);
    """
    name = "rvv.vwadd_vx_i16m2Op"
    vs = operand_def(RVVInt8M1Type)
    scalar = operand_def(IntegerType)
    avl = operand_def(IndexType)
    result = result_def(RVVInt16M2Type)
    def __init__(self, vs: SSAValue, scalar: SSAValue, avl: SSAValue):
        super().__init__(
            operands=[vs, scalar, avl],
            result_types=[RVVInt16M2Type()]
        )


@irdl_op_definition
class vmv_v_x_i32m4Op(IRDLOperation):
    """
    C00 = __riscv_vmv_v_x_i32m4(0, vlength);
    """
    name = "rvv.vmv_v_x_i32m4Op"
    scalar = operand_def(IntegerType)
    avl = operand_def(IndexType)
    result = result_def(RVVInt32M4Type)
    def __init__(self, scalar: SSAValue, avl: SSAValue):
        super().__init__(
            operands= [scalar, avl], result_types=[RVVInt32M4Type()]
        )

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
    def __init__(self, vd: SSAValue, scalar: SSAValue, vs2: SSAValue, avl: SSAValue):
        super().__init__(operands=[vd, scalar, vs2, avl], result_types=[RVVInt32M4Type()])

@irdl_op_definition
class vse32_v_i32m4Op(IRDLOperation):
    """__riscv_vse32_v_i32m4(ptr, vd, vl) → ()
      __riscv_vse32_v_i32m4(&Ccol(0, 0), C00, vlength);
    """
    name = "rvv.vse32_v_i32m4"
    memref = operand_def(MemRefType)
    offset = operand_def(IndexType)
    vec    = operand_def(RVVInt32M4Type)
    avl    = operand_def(IndexType)
    def __init__(self, memref: SSAValue, offset: SSAValue, vec: SSAValue, avl: SSAValue):
        super().__init__(operands=[memref, offset, vec, avl], result_types=[])


@irdl_op_definition
class vfmacc_vf_f32m1Op(IRDLOperation):
    """
    Fused multiply-accumulate: vd = vd + memref[offset] * vs
    C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0, (vl));
    """
    name = "rvv.vfmacc_vf_f32m1Op"
    vd = operand_def(RVVFloat32M1Type)
    memref = operand_def(MemRefType)
    offset = operand_def(IndexType)
    vs = operand_def(RVVFloat32M1Type)
    avl = operand_def(IndexType)

    result = result_def(RVVFloat32M1Type)

    def __init__(
        self,
        vd: SSAValue,
        memref: SSAValue,
        offset: SSAValue,
        vs: SSAValue,
        avl: SSAValue,
    ):
        super().__init__(
            operands=[vd, memref, offset, vs, avl],
            result_types=[RVVFloat32M1Type()],  # No parameters!
        )


@irdl_op_definition
class vfmv_v_f_f32m1Op(IRDLOperation):
    """
    C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f, (vl));
    """

    name = "rvv.vfmv_v_f_f32m1"

    scalar = operand_def(Float32Type)
    avl = operand_def(IndexType)

    result = result_def(RVVFloat32M1Type)

    def __init__(self, scalar: SSAValue, avl: SSAValue):
        super().__init__(operands=[scalar, avl], result_types=[RVVFloat32M1Type()])


@irdl_op_definition
class vse32_v_f32m1Op(IRDLOperation):
    """
    Implement    __riscv_vse32_v_f32m1(&C[0], C_reg_0_0, (vl));
    """

    name = "rvv.vse32_v_f32m1Op"

    memref = operand_def(MemRefType)
    offset = operand_def(IndexType)
    vec = operand_def(RVVFloat32M1Type)
    avl = operand_def(IndexType)

    def __init__(
        self, memref: SSAValue, offset: SSAValue, vector: SSAValue, avl: SSAValue
    ):
        super().__init__(
            operands=[memref, offset, vector, avl],
            result_types=[],  # Empty, this func dont return anything
        )

# python -c "from xdsltemplate.dialects.rvv import vle8_v_i8m1Op, RVVInt8M1Type; print('OK')"
RVV = Dialect(
    "rvv",
    [
        SetvlOp,
        # FLOAT
        vle32_v_f32m1Op,
        vfmacc_vf_f32m1Op,
        vfmv_v_f_f32m1Op,
        vse32_v_f32m1Op,
        # INT
        vle8_v_i8m1Op,
        vle32_v_i32m4Op,
        vwadd_vx_i16m2Op,
        vmv_v_x_i32m4Op,
        vwmacc_vx_i32m4Op,
        vse32_v_i32m4Op,
    ],
    [
        # FLOAT
        RVVvectorType,
        RVVFloat32M1Type,
        # INT
        RVVInt8M1Type,
        RVVInt16M2Type,
        RVVInt32M4Type,
    ],
)
