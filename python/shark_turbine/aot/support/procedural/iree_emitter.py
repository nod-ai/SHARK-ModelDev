# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python API for IREE's high-level tensor dialects."""

from typing import Any, List, Sequence, Tuple, Union

import functools

import torch

from ..ir_imports import (
    IndexType,
    RankedTensorType,
    StringAttr,
    Value,
    flow_d,
)

from ..ir_utils import (
    TORCH_DTYPE_TO_IREE_TYPE,
    build_index_value,
)

from .base import (
    Intrinsic,
    IrValueTensor,
    IrValueScalar,
    current_ir_trace,
    ShapedTypeDynamicSizeSentinel,
)

BuildableScalarValue = Union[IrValueScalar, Value]
BuildableTensorDimDecl = Union[int, Value]
BuildableTensorType = IrValueTensor
BuildableIndexType = Union[Value, int]
StaticIndexType = int


def cast_scalar_value(x: BuildableScalarValue) -> Value:
    x = unwrap_intrinsic_value(x)
    if not isinstance(x, Value):
        raise ValueError(f"Expected a scalar value but got {x}")
    return x


def cast_tensor_value(x: BuildableTensorType) -> IrValueTensor:
    assert isinstance(x, IrValueTensor), f"Expected a tensor but got {type(x)}"
    return x


def cast_index_value(x: BuildableIndexType) -> Value:
    if isinstance(x, int):
        return build_index_value(x)
    else:
        return x


def cast_static_bounded_index(x: int, min_value: int, max_value: int) -> int:
    if not isinstance(x, int):
        raise ValueError(f"Expected int but got {type(x)}")
    if x < min_value or x > max_value:
        raise ValueError(
            f"Expected int in range [{min_value}, {max_value}] but got {x}"
        )
    return x


def cast_tensor_dim_decl(
    xs: Sequence[BuildableTensorDimDecl],
) -> Tuple[Sequence[int], Sequence[Value]]:
    """Casts a sequence of tensor declaration dimensions to dims suitable
    for construction of a TensorType and a sequence of dynamic dim values."""
    dim_decls: List[int] = []
    dynamic_dim_values: List[Value] = []
    for x in xs:
        x = unwrap_intrinsic_value(x)
        if isinstance(x, Value):
            assert_value_is_index(x)
            dim_decls.append(ShapedTypeDynamicSizeSentinel)
            dynamic_dim_values.append(x)
        elif isinstance(x, int) and x >= 0:
            dim_decls.append(x)
        else:
            raise ValueError(
                f"Expected a tensor dimension as a positive integer or None but got {x}"
            )
    return dim_decls, dynamic_dim_values


def assert_value_is_index(x: Value):
    t = x.type
    if not IndexType.isinstance(t):
        raise ValueError(f"Expected an index value but got {t}")


def unwrap_intrinsic_value(x) -> Any:
    if isinstance(x, Intrinsic):
        x, *rest = x.resolve_ir_values(current_ir_trace())
        if rest:
            raise ValueError(
                f"Expected a value that has an arity of one component but for {len(rest) + 1}"
            )
    return x


def emitter(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        t = current_ir_trace()
        with t.loc, t.ip:
            return f(*args, **kwargs)

    return wrapper


class IREEEmitter:
    @emitter
    def tensor_dim(self, source: BuildableTensorType, index: int) -> "IrValueScalar":
        """Gets the dimension size of a tensor at a static position."""
        source = cast_tensor_value(source)
        index = cast_static_bounded_index(index, 0, source.rank - 1)
        return IrValueScalar(source.get_dim_value(index))

    @emitter
    def tensor_empty(
        self, *dims: BuildableTensorDimDecl, dtype: torch.dtype = torch.float32
    ) -> IrValueTensor:
        """Constructs a tensor with uninitialized values.

        TODO: Support an IREE/raw element type in addition to the torch dtype.
        """
        dim_decls, dyn_dim_values = cast_tensor_dim_decl(dims)
        try:
            element_type = TORCH_DTYPE_TO_IREE_TYPE[dtype]()
        except KeyError:
            raise ValueError(f"Could not map Torch dtype {dtype} to an IREE type")
        tensor_type = RankedTensorType.get(dim_decls, element_type)
        raw_tensor = flow_d.TensorEmptyOp(tensor_type, dyn_dim_values).result
        result = IrValueTensor(raw_tensor, dtype=dtype)
        result.set_dynamic_dim_values(dyn_dim_values)
        return result

    @emitter
    def tensor_splat(
        self,
        *dims: BuildableTensorDimDecl,
        value: BuildableScalarValue,
        dtype: torch.dtype,
    ) -> "IrValueTensor":
        # TODO: Type infer the dtype if missing.
        dim_decls, dyn_dim_values = cast_tensor_dim_decl(dims)
        try:
            element_type = TORCH_DTYPE_TO_IREE_TYPE[dtype]()
        except KeyError:
            raise ValueError(f"Could not map Torch dtype {dtype} to an IREE type")
        value = cast_scalar_value(value)
        if value.type != element_type:
            raise ValueError(
                f"Provided splat value ({type(value)}) does not match dtype {dtype}"
            )
        tensor_type = RankedTensorType.get(dim_decls, element_type)
        raw_tensor = flow_d.TensorSplatOp(tensor_type, value, dyn_dim_values).result
        result = IrValueTensor(raw_tensor, dtype=dtype)
        result.set_dynamic_dim_values(dyn_dim_values)
        return result

    @emitter
    def tensor_trace(self, key: str, *ts: BuildableTensorType):
        ts = [cast_tensor_value(t).ir_value for t in ts]
        flow_d.TensorTraceOp(StringAttr.get(key), ts)


# Circular imports to resolve typing.
from .primitives import (
    IrValueScalar,
    IrValueTensor,
)
