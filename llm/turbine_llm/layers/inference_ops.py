# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Operations involving InferenceTensors.

This includes a light-weight type switching mechanism that lets us pretend
these are normal operations under certain circumstances.

There are really only a handful of operations that are ever done on packed
Inference tensors, and we support those here.
"""

from typing import Union

import torch

from .base import (
    InferenceTensor,
    PrimitiveTensor,
    QuantizedTensor,
)


def matmul(
    lhs: torch.Tensor,
    rhs: Union[torch.Tensor, InferenceTensor],
    *,
    transpose_rhs: bool = True,
) -> torch.Tensor:
    """Performs a matmul where the RHS may be an InferenceTensor.

    Unlike torch.matmul, this variant is optimized for emission of a fused
    `matmul(lhs, rhs.T)` and the `transpose_rhs=` defaults to True, indicating
    the the RHS is expected to have been transposed already (by some outside
    force). Most inference optimizers will store their weights in this way
    and assume fusions that operate on them, so we just make it the default.

    Args:
      lhs: Left hand side tensor. Can have dimensionality > 2 for batch.
      rhs: Right hand side tensor.
      transpose_rhs: Whether the right hand side should be transposed prior
        to matmul.
    """
    if transpose_rhs:
        assert (
            len(rhs.shape) == 2
        ), f"Expected 2d rhs for transpose_rhs=True. Got: {rhs.shape}"

    if isinstance(rhs, QuantizedTensor):
        # By default, unpack and dequantize the rhs. This produces correct results
        # for Torch but is likely not the right thing for anything else.
        # TODO: Consult a dispatch table for the engine-specific op to use here.
        rhs_torch = rhs.unpack().dequant(lhs.dtype)
        return _matmul_torch(
            lhs,
            rhs_torch,
            transpose_rhs=transpose_rhs,
        )
    elif isinstance(rhs, PrimitiveTensor):
        # Convertible to a Torch tensor without custom layout.
        rhs_torch = rhs.as_torch()
        return _matmul_torch(
            lhs,
            rhs_torch,
            transpose_rhs=transpose_rhs,
        )
    else:
        # Treat it as a torch Tensor.
        assert isinstance(rhs, torch.Tensor)
        return _matmul_torch(lhs, rhs, transpose_rhs=transpose_rhs)


def _matmul_torch(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    transpose_rhs: bool,
):
    if transpose_rhs:
        rhs = rhs.T
    return torch.matmul(lhs, rhs)
