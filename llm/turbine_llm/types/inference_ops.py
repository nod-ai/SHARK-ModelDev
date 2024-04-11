# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Union

import torch
import torch.nn.functional as F

from .tensors import InferenceTensor, PrimitiveTensor, QuantizedTensor

__all__ = [
    "InferenceOps",
]


class InferenceOps:
    """Operations involving InferenceTensors.

    There are really only a handful of operations that are ever done on packed
    Inference tensors, and we support those here on a default class with a
    PyTorch whole-tensor based implementation. The default implementation should
    be correct but can be swapped for more layout/target sensitive subclasses as
    desired.

    The InferenceOps class can be accessed on any Theta object, which also
    provides a single place where it can be customized.
    """

    def embedding_lookup(
        self,
        input: torch.Tensor,
        embedding_matrix: Union[torch.Tensor, InferenceTensor],
        dtype: torch.dtype,
    ):
        """Performs the equivalent of F.embedding(input, embedding_matrix).

        Note that the default algorithm will unquantize the embedding_matrix to
        do the lookup, which is inefficient. Specializations should decompose
        this as appropriate for quantized arithmetic.
        """
        if isinstance(embedding_matrix, InferenceTensor):
            if isinstance(embedding_matrix, QuantizedTensor):
                embedding_matrix = embedding_matrix.unpack().dequant(dtype)
            elif isinstance(embedding_matrix, PrimitiveTensor):
                embedding_matrix = embedding_matrix.as_torch().to(dtype)
            else:
                raise AssertionError(
                    f"Unsupported InferenceTensor: {type(embedding_matrix)}"
                )
        return F.embedding(input, embedding_matrix)  # type: ignore

    def matmul(
        self,
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
        rhs: Right hand side tensor. Must be 2d.
        transpose_rhs: Whether the right hand side should be transposed prior
            to matmul.
        """
        assert len(rhs.shape) == 2, f"Expected 2d matmul rhs for. Got: {rhs.shape}"

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
            rhs_torch = rhs.as_torch(dtype=lhs.dtype)
            return _matmul_torch(
                lhs,
                rhs_torch,
                transpose_rhs=transpose_rhs,
            )
        else:
            # Treat it as a torch Tensor.
            assert isinstance(rhs, torch.Tensor)
            return _matmul_torch(lhs, rhs, transpose_rhs=transpose_rhs)

    def rms_norm(
        self,
        x: torch.Tensor,
        weight: Union[torch.Tensor, InferenceTensor],
        *,
        epsilon: float,
    ):
        """Computes the full, unbiased RMS normalization of an input."""
        if isinstance(weight, InferenceTensor):
            if isinstance(weight, QuantizedTensor):
                weight = weight.unpack().dequant(x.dtype)
            elif isinstance(weight, PrimitiveTensor):
                weight = weight.as_torch()
            else:
                raise AssertionError(f"Unsupported InferenceTensor: {type(weight)}")
        variance = x.pow(2).mean(-1, keepdim=True)
        output = x * torch.rsqrt(variance + epsilon)
        output = output * weight
        return output


def _matmul_torch(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    transpose_rhs: bool,
):
    if transpose_rhs:
        rhs = rhs.T
    return torch.matmul(lhs, rhs)
