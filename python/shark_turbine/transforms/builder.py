# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Optional

from ..dynamo import type_conversion

from iree.compiler.ir import (
    Context,
    RankedTensorType,
    Type as IrType,
)

__all__ = [
    "Builder",
]


class Builder:
    def __init__(self, context: Context = None):
        if not context:
            context = Context.current
        self.context = context
        self.native_type_conversion = type_conversion.NativeTypeConverter(self.context)

    def to_native_type(self, t: IrType) -> IrType:
        return self.native_type_conversion.torch_type_to_native(t)

    def to_native_tensor_type(self, t: IrType) -> RankedTensorType:
        if not RankedTensorType.isinstance(t):
            try:
                return RankedTensorType(self.to_native_type(t))
            except Exception as e:
                raise ValueError(f"Could not convert to tensor type ({t})") from e
        return RankedTensorType(t)

    def get_tensor_dims(self, tensor_type: IrType) -> List[Optional[int]]:
        rt = self.to_native_tensor_type(tensor_type)
        return [
            None if rt.is_dynamic_dim(axis) else rt.get_dim_size(axis)
            for axis in range(rt.rank)
        ]

    def get_tensor_element_type(self, tensor_type: IrType) -> IrType:
        rt = self.to_native_tensor_type(tensor_type)
        return rt.element_type
