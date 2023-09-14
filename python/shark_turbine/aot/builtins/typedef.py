# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builtins for annotating types."""

from typing import Any, Optional

import torch

from iree.compiler.ir import (
    RankedTensorType,
    Type as IrType,
    Value,
)

from ..builder import (
    ModuleBuilder,
)

from ..procedural import (
    AbstractIntrinsic,
    Intrinsic,
    IrValueTensor,
    Location,
)


TORCH_DTYPE_TO_IREE_TYPE_ASM = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.uint8: "i8",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "i1",
    torch.qint8: "i8",
    torch.quint8: "i8",
    torch.complex32: "complex<f16>",
    torch.complex64: "complex<f32>",
    torch.complex128: "complex<f64>",
}


class AbstractTensor(AbstractIntrinsic):
    """Represents a tensor of known rank and dtype."""

    __slots__ = [
        "size",
        "dtype",
    ]

    def __init__(self, *size: Optional[int], dtype: torch.dtype = torch.float32):
        self.size = tuple(size)
        self.dtype = dtype

    def __repr__(self):
        return f"AbstractTensor({', '.join(str(s) for s in self.size)}, dtype={self.dtype})"

    def create_intrinsic(self, ir_value: Value) -> Intrinsic:
        return IrValueTensor(ir_value, self.dtype)

    def get_ir_type(self, builder: ModuleBuilder) -> IrType:
        element_type = builder.torch_dtype_to_iree_type(self.dtype)
        with Location.unknown(builder.context):
            tensor_type = RankedTensorType.get(
                [s if s is not None else -1 for s in self.size], element_type
            )
        return tensor_type
