# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Type, Union

from types import NotImplementedType

import torch
import torch.nn.functional as F

from ..types import (
    BaseInferenceOps,
    BlockScaledLayout,
    InferenceTensor,
    PrimitiveTensor,
    QuantizedTensor,
    gguf_interop,
)

from .matmul import mmtfp, mmt_block_scaled_q8

__all__ = [
    "BaseInferenceOps",
]


class CustomInferenceOps(BaseInferenceOps):
    """Default implementation that does custom dispatch of accelerated mixed
    precision cases.
    """

    def _matmul(
        self,
        lhs: torch.Tensor,
        rhs: Union[torch.Tensor, InferenceTensor],
        *,
        transpose_rhs: bool = True,
    ) -> Union[NotImplementedType, torch.Tensor]:
        # We only accelerate matmuls with transposed RHS set up for inference
        # ... like civilized people.
        if not transpose_rhs or not isinstance(rhs, InferenceTensor):
            return NotImplemented
        if len(lhs.shape) > 3:
            # Only 2d or 3d batch matmuls currently supported.
            return NotImplemented

        if isinstance(rhs, PrimitiveTensor):
            return mmtfp(lhs, rhs.as_torch())

        if not isinstance(rhs, QuantizedTensor):
            return NotImplemented

        # Handle quantized tensor layout switched.
        handler = _QMMT_DISPATCH.get(type(rhs))
        if handler is None:
            return NotImplemented
        return handler(lhs, rhs)


def _mmt_block_scaled(lhs: torch.Tensor, rhs: QuantizedTensor[BlockScaledLayout]):
    """Generic fallback kernel for block scaled layouts.

    This will unpack and operate generically on planar scales/blocks vs a packed
    struct. This may be fine for certain platforms but there is micro-optimization
    potential if specializing further to the packed layout.
    """
    rhs_unpacked = rhs.unpack()
    return mmt_block_scaled_q8(lhs, rhs_unpacked.d, rhs_unpacked.qs)


_QMMT_DISPATCH: dict[type, Callable] = {gguf_interop.Q8_0: _mmt_block_scaled}
