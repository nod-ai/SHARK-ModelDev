from typing import Any, Optional, Union

from .._support.indexing import (
    IndexExpr,
    SymIndex,
)

from .base import (
    CodegenError,
    NDEBUG,
)

from .ir import (
    Attribute,
    Context,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    IrType,
    Location,
    Operation,
    SymbolTable,
    Value,
    VectorType,
    arith_d,
    math_d,
    builtin_d,
)


# TODO: Have a way upstream to check if a floating point type.
FLOAT_TYPES_ASM = {
    "bf16",
    "f16",
    "f32",
    "f64",
    # TODO: FP8 types.
}


class IRProxyValue:
    """Wrapper around an (ir.Value, py_value) for handling notionally python
    proxies that are associated with an IR Value.
    """

    __slots__ = [
        "ir_value",
        "py_value",
    ]

    def __init__(self, ir_value: Value, py_value: Any = None):
        self.ir_value = ir_value
        self.py_value = py_value
        assert NDEBUG or self.validate()

    def validate(self):
        assert isinstance(self.ir_value, Value), f"Got {type(self.ir_value)}"
        return True

    def __repr__(self):
        return f"<IRProxyValue({self.ir_value}):{self.py_value}>"


class ModuleBuilder:
    def __init__(
        self,
        *,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ):
        if module_op:
            self.module_op = module_op
            self.body_block = module_op.regions[0].blocks[0]
        else:
            if not context:
                context = Context()
            self.module_op = builtin_d.ModuleOp(loc=Location.unknown(context))
            self.body_block = self.module_op.body
        self.context = self.module_op.context
        self.unknown_loc = Location.unknown(self.context)
        self.symbol_table = SymbolTable(self.module_op)


class _ScalarBuilder:
    def is_floating_point_type(self, t: IrType) -> bool:
        return str(t) in FLOAT_TYPES_ASM

    def is_integer_type(self, t: IrType) -> bool:
        return IntegerType.isinstance(t)

    def promote(self, value: Value, to_type: IrType) -> Value:
        value_type = value.type
        # Short-circuit if already the right type.
        if value_type == to_type:
            return value

        attr_name = f"promote_{value_type}_to_{to_type}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"No implemented path to implicitly promote scalar `{value_type}` to `{to_type}` (tried '{attr_name}')"
            )
        return handler(value, to_type)

    def zero_attr(self, t: IrType) -> Attribute:
        attr_name = f"zero_attr_{t}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot derive a zero value for type `{t}` (tried '{attr_name}')"
            )
        return handler(t)

    def constant(self, py_value) -> IRProxyValue:
        attr_name = f"py_constant_{type(py_value).__name__}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot convert Python value to constant: {py_value} of type {type(py_value)} (tried '{attr_name}')"
            )
        return handler(py_value)

    def binary_arithmetic(
        self, op: str, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        lhs_ir_type = lhs.ir_value.type
        rhs_ir_type = rhs.ir_value.type
        attr_name = f"binary_{op}_{lhs_ir_type}_{rhs_ir_type}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot perform binary arithmetic operation '{op}' between {lhs_ir_type} and {rhs_ir_type} (tried '{attr_name}')"
            )
        return handler(lhs, rhs)

    def binary_vector_arithmetic(
        self, op: str, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        lhs_ir = lhs.ir_value
        rhs_ir = rhs.ir_value
        lhs_element_type = VectorType(lhs_ir.type).element_type
        rhs_element_type = VectorType(rhs_ir.type).element_type
        attr_name = f"binary_{op}_{lhs_element_type}_{rhs_element_type}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot perform binary arithmetic operation '{op}' between {lhs_ir.type} and {rhs_ir.type} (tried '{attr_name}')"
            )
        return handler(lhs, rhs)

    def unary_arithmetic(self, op: str, val: IRProxyValue) -> IRProxyValue:
        val_ir_type = val.ir_value.type
        attr_name = f"unary_{op}_{val_ir_type}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot perform unary arithmetic operation '{op}' on {val_ir_type} (tried '{attr_name}')"
            )
        return handler(val)

    def unary_vector_arithmetic(self, op: str, val: IRProxyValue) -> IRProxyValue:
        val_ir = val.ir_value
        val_element_type = VectorType(val_ir.type).element_type
        attr_name = f"unary_{op}_{val_element_type}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot perform unary arithmetic operation '{op}' on {val_ir.type} (tried '{attr_name}')"
            )
        return handler(val)

    def promote_index_to_f32(self, value: Value, to_type: IrType) -> Value:
        i32_type = IntegerType.get_signless(32)
        i32 = arith_d.index_cast(i32_type, value)
        return arith_d.sitofp(to_type, i32)

    def zero_attr_f32(self, t: IrType) -> Attribute:
        return FloatAttr.get(t, 0.0)

    def py_constant_int(self, py_value) -> IRProxyValue:
        # If coming from a stock 'int' Python type with no idea how to convert it,
        # there isn't much smart we can do. We conservatively treat 'index' as
        # reasonable.
        result_type = IndexType.get()
        return IRProxyValue(
            arith_d.constant(result_type, IntegerAttr.get(result_type, py_value)),
            py_value,
        )

    # Binary index arithmetic.
    def binary_add_index_index(
        self, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        return IRProxyValue(arith_d.addi(lhs.ir_value, rhs.ir_value))

    def binary_mul_index_index(
        self, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        return IRProxyValue(arith_d.muli(lhs.ir_value, rhs.ir_value))

    def binary_sub_index_index(
        self, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        return IRProxyValue(arith_d.subi(lhs.ir_value, rhs.ir_value))

    def binary_mod_index_index(
        self, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        return IRProxyValue(arith_d.remsi(lhs.ir_value, rhs.ir_value))

    def binary_floordiv_index_index(
        self, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        return IRProxyValue(arith_d.floordivsi(lhs.ir_value, rhs.ir_value))

    # Binary f32 arithmetic
    def binary_add_f32_f32(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.addf(lhs.ir_value, rhs.ir_value))

    def binary_mul_f32_f32(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.mulf(lhs.ir_value, rhs.ir_value))

    def binary_sub_f32_f32(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.subf(lhs.ir_value, rhs.ir_value))

    def binary_mod_f32_f32(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.remf(lhs.ir_value, rhs.ir_value))

    def binary_truediv_f32_f32(
        self, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        return IRProxyValue(arith_d.divf(lhs.ir_value, rhs.ir_value))

    def unary_exp2_f32(self, val: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(math_d.exp2(val.ir_value))


ScalarBuilder = _ScalarBuilder()
