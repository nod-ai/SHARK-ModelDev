# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional, Union, Collection

from pathlib import Path

from types import NotImplementedType
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from shark_turbine.aot import ParameterArchiveBuilder

from .tensors import InferenceTensor, PrimitiveTensor, QuantizedTensor

__all__ = [
    "BaseInferenceOps",
    "Dataset",
    "Theta",
]

IOReportCallback = Callable[[str], None]


class Theta:
    """Subset of parameter tensors used for inference."""

    def __init__(
        self,
        tensors: dict,
        *,
        ops: Optional["BaseInferenceOps"] = None,
        already_nested: bool = False,
    ):
        self._tensors = tensors if already_nested else _flat_to_nested_dict(tensors)
        if ops is None:
            # Use the custom op library by default. Note that since the ops
            # namespace depends on types, we have to lazy load it.
            from ..ops import CustomInferenceOps

            ops = CustomInferenceOps()
        self.ops = ops

    def flatten(self) -> dict[str, InferenceTensor]:
        results = {}

        def accum(prefix, child):
            for key, value in child.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    accum(new_prefix, value)
                else:
                    results[new_prefix] = value

        accum("", self._tensors)
        return results

    def tensor(self, *name_path: Union[str, int]) -> InferenceTensor:
        current_ts = self._tensors
        try:
            for part in name_path[0:-1]:
                current_ts = current_ts[str(part)]
            last = name_path[-1]
            t = current_ts[str(last)]
        except KeyError:
            raise KeyError(
                f"Unknown parameter {name_path} (in Theta object "
                f"containing {self.keys})"
            )
        return t

    @property
    def keys(self) -> Collection[str]:
        return self._tensors.keys()

    @property
    def tensors(self) -> Collection[InferenceTensor]:
        return self._tensors.values()

    def __call__(self, *name_path: Union[str, int]) -> "Theta":
        current_ts = self._tensors
        try:
            for part in name_path:
                current_ts = current_ts[str(part)]
        except KeyError:
            raise KeyError(f"Sub-theta {name_path} not found")
        return Theta(current_ts, ops=self.ops)

    def __repr__(self):
        return f"Theta({self.keys})"

    def add_tensors_to_archive(
        self,
        irpa: ParameterArchiveBuilder,
        *,
        io_report_callback: Optional[IOReportCallback] = None,
    ):
        """Adds tensors to the given archive builder."""
        for inference_tensor in self.flatten().values():
            if io_report_callback:
                io_report_callback(f"Add {inference_tensor}")
            inference_tensor.add_to_archive(irpa)


def _flat_to_nested_dict(flat: dict[str, Any]) -> dict[str, Any]:
    nested: dict = {}

    def add_to_dict(
        name: str,
        value,
    ):
        current = nested

        parts = name.split(".")
        for part in parts[0:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            assert isinstance(
                current, dict
            ), f"Name collision in parameter dict: {name}"
        current[parts[-1]] = value

    for name, value in flat.items():
        add_to_dict(name, value)
    return nested


@dataclass
class Dataset:
    """Top level configuration for a model.

    This consists of:

    * Dataset level hyper-parameters (properties).
    * Root theta with materialized parameters (Theta).
    """

    properties: dict[str, Any]
    root_theta: Theta

    def save(
        self,
        path: Union[str, Path],
        *,
        io_report_callback: Optional[IOReportCallback] = None,
    ):
        builder = ParameterArchiveBuilder()
        self.root_theta.add_tensors_to_archive(
            builder, io_report_callback=io_report_callback
        )
        if io_report_callback:
            io_report_callback("Saving file")
            builder.save(path)


class BaseInferenceOps:
    """Operations involving InferenceTensors.

    There are really only a handful of operations that are ever done on packed
    Inference tensors, and we support those here on a default class with a
    PyTorch whole-tensor based implementation. The default implementation should
    be correct but can be swapped for more layout/target sensitive subclasses as
    desired.

    The InferenceOps class can be accessed on any Theta object, which also
    provides a single place where it can be customized.

    This class was designed to be subclassed. Ops will generally attempt to
    dispatch to a private function of the same name (with leading underscore)
    and will only execute the default Torch implementation if it returns
    NotImplemented.
    """

    def embedding_lookup(
        self,
        input: torch.Tensor,
        embedding_matrix: Union[torch.Tensor, InferenceTensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Performs the equivalent of F.embedding(input, embedding_matrix).

        Note that the default algorithm will unquantize the embedding_matrix to
        do the lookup, which is inefficient. Specializations should decompose
        this as appropriate for quantized arithmetic.
        """
        delegated = self._embedding_lookup(input, embedding_matrix, dtype)
        if delegated is not NotImplemented:
            return delegated
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

    def _embedding_lookup(
        self,
        input: torch.Tensor,
        embedding_matrix: Union[torch.Tensor, InferenceTensor],
        dtype: torch.dtype,
    ) -> Union[NotImplementedType, torch.Tensor]:
        return NotImplemented

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
        delegated = self._matmul(lhs, rhs, transpose_rhs=transpose_rhs)
        if delegated is not NotImplemented:
            return delegated

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

    def _matmul(
        self,
        lhs: torch.Tensor,
        rhs: Union[torch.Tensor, InferenceTensor],
        *,
        transpose_rhs: bool = True,
    ) -> Union[NotImplementedType, torch.Tensor]:
        return NotImplemented

    def rms_norm(
        self,
        x: torch.Tensor,
        weight: Union[torch.Tensor, InferenceTensor],
        *,
        epsilon: float,
    ) -> torch.Tensor:
        """Computes the full, unbiased RMS normalization of an input."""
        delegated = self._rms_norm(x, weight, epsilon=epsilon)
        if delegated is not NotImplemented:
            return delegated
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

    def _rms_norm(
        self,
        x: torch.Tensor,
        weight: Union[torch.Tensor, InferenceTensor],
        *,
        epsilon: float,
    ) -> Union[NotImplementedType, torch.Tensor]:
        return NotImplemented


def _matmul_torch(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    transpose_rhs: bool,
):
    if transpose_rhs:
        rhs = rhs.T
    return torch.matmul(lhs, rhs.to(lhs.dtype))
