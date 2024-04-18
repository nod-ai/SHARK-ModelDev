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
    BlockScaledI4Layout,
    InferenceTensor,
    PrimitiveTensor,
    QuantizedTensor,
    SuperBlockOffsetScaled_4_6_Layout,
    gguf_interop,
)

from .matmul import (
    mmtfp,
    mmt_block_scaled_offset_q4_unsigned,
    mmt_block_scaled_q8,
    mmt_super_block_scaled_offset_q4_unsigned,
)

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
        handler = _QMMT_DISPATCH.get(rhs.layout_type)
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
    assert rhs_unpacked.m is None, "NYI: Q8 block scaled with offset"
    return mmt_block_scaled_q8(lhs, rhs_unpacked.d, rhs_unpacked.qs)


def _mmt_block_scaled_q4(lhs: torch.Tensor, rhs: QuantizedTensor[BlockScaledI4Layout]):
    """Generic fallback kernel for an unsigned, block scaled Q4."""
    rhs_unpacked = rhs.unpack()
    assert rhs_unpacked.m is not None, "NYI: Q4 without offset not"
    assert not rhs_unpacked.signed, "NYI: Q4 signed"
    return mmt_block_scaled_offset_q4_unsigned(
        a=lhs, d=rhs_unpacked.d, qs=rhs_unpacked.qs_bit_packed, m=rhs_unpacked.m
    )


def _mmt_super_block_offset_scaled_4_6_q4(
    lhs: torch.Tensor, rhs: QuantizedTensor[SuperBlockOffsetScaled_4_6_Layout]
):
    rhs_unpacked = rhs.unpack()
    sb_scales_hi, sb_scales_low = rhs_unpacked.sb_scales_bit_packed
    sb_mins_hi, sb_mins_low = rhs_unpacked.sb_mins_bit_packed
    return mmt_super_block_scaled_offset_q4_unsigned(
        lhs,
        rhs_unpacked.d,
        rhs_unpacked.dmin,
        sb_scales_hi,
        sb_scales_low,
        sb_mins_hi,
        sb_mins_low,
        rhs_unpacked.qs_bit_packed,
    )


_QMMT_DISPATCH: dict[type, Callable] = {
    BlockScaledI4Layout: _mmt_block_scaled_q4,
    BlockScaledLayout: _mmt_block_scaled,
    SuperBlockOffsetScaled_4_6_Layout: _mmt_super_block_offset_scaled_4_6_q4,
}
