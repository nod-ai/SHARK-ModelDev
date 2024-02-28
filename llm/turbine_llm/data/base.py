# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Collection, TypeVar, Generic, Type
from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = [
    "Dataset",
    "InferenceTensor",
    "PrimitiveTensor",
    "QuantizedTensor",
    "Theta",
    "QuantizedLayout",
]


class QuantizedLayout(ABC):
    @abstractmethod
    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        ...


QuantizedLayoutT = TypeVar("QuantizedLayoutT", bound=QuantizedLayout)


class InferenceTensor(ABC):
    """Provides access to a tensor in the model used for inference.

    InferenceTensors have a richer structure than "normal" training tensors
    since they often involve a degree of layout on top of the raw data tensor.
    """

    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape

    @property
    @abstractmethod
    def globals(self) -> dict[str, torch.Tensor]:
        """Returns a mapping of global name to root tensor.

        The primary accessors on an InferenceTensor access the root tensors in
        the global set, all of which in a root Theta must have unique names.
        """
        ...


class PrimitiveTensor(InferenceTensor):
    """An InferenceTensor without any kind of special layout.

    These can be directly operated on as a torch.Tensor.
    """

    @abstractmethod
    def as_torch(self) -> torch.Tensor:
        """Accesses the raw data as a torch tensor.

        If the tensor is packed in some way, this may bare no resemblance to
        the logical arrangement of the data.
        """
        ...


class DefaultPrimitiveTensor(PrimitiveTensor):
    """Concrete implementation of a PrimitiveTensor based on a single tensor."""

    def __init__(self, name: str, data: torch.Tensor):
        super().__init__(name, list(data.shape))
        self._data = data

    def as_torch(self) -> torch.Tensor:
        return self._data

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        return {
            self.name: self._data,
        }

    def __repr__(self):
        return f"PrimitiveTensor({self.name}, {self.shape})"


class QuantizedTensor(InferenceTensor, Generic[QuantizedLayoutT]):
    """An inference tensor that is quantized/packed."""

    def __init__(
        self,
        name: str,
        shape: list[int],
        *,
        layout_type: Type[QuantizedLayout],
    ):
        super().__init__(name, shape)
        self.layout_type = layout_type

    @abstractmethod
    def unpack(self) -> QuantizedLayoutT:
        ...


class Theta:
    """Subset of parameter tensors used for inference."""

    def __init__(
        self,
        tensors: dict,
        *,
        ops: Optional["InferenceOps"] = None,
        already_nested: bool = False,
    ):
        self._tensors = tensors if already_nested else _flat_to_nested_dict(tensors)
        self.ops = ops if ops is not None else InferenceOps()

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
