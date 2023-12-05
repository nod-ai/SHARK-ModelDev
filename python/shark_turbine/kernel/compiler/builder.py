from typing import Optional

from .base import (
    CodegenError,
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
    Value,
    arith_d,
    builtin_d,
)


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


class _ScalarBuilder:
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

    def constant(self, py_value) -> Value:
        attr_name = f"py_constant_{type(py_value).__name__}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot convert Python value to constant: {py_value} of type {type(py_value)} (tried '{attr_name}')"
            )
        return handler(py_value)

    def binary_arithmetic(self, op: str, lhs: Value, rhs: Value) -> Value:
        attr_name = f"binary_{op}_{lhs.type}_{rhs.type}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot perform binary arithmetic operation '{op}' between {lhs.type} and {rhs.type} (tried '{attr_name}')"
            )
        return handler(lhs, rhs)

    def promote_index_to_f32(self, value: Value, to_type: IrType) -> Value:
        i32_type = IntegerType.get_signless(32)
        i32 = arith_d.index_cast(i32_type, value)
        return arith_d.sitofp(to_type, i32)

    def zero_attr_f32(self, t: IrType) -> Attribute:
        return FloatAttr.get(t, 0.0)

    def py_constant_int(self, py_value) -> Value:
        # If coming from a stock 'int' Python type with no idea how to convert it,
        # there isn't much smart we can do. We conservatively treat 'index' as
        # reasonable.
        attr = IntegerAttr.get(IndexType.get(), py_value)
        return arith_d.constant(attr)

    def binary_add_index_index(self, lhs: Value, rhs: Value) -> Value:
        return arith_d.addi(lhs, rhs)

    def binary_mul_index_index(self, lhs: Value, rhs: Value) -> Value:
        return arith_d.muli(lhs, rhs)

    def binary_sub_index_index(self, lhs: Value, rhs: Value) -> Value:
        return arith_d.subi(lhs, rhs)

    def binary_mod_index_index(self, lhs: Value, rhs: Value) -> Value:
        return arith_d.remsi(lhs, rhs)

    def binary_floordiv_index_index(self, lhs: Value, rhs: Value) -> Value:
        return arith_d.floordivsi(lhs, rhs)


ScalarBuilder = _ScalarBuilder()
