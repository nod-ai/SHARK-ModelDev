# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Optional, Union, Collection, Iterator, TypeVar, Generic, Type
from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = [
    "InferenceTensor",
    "PrimitiveTensor",
    "QuantizedTensor",
    "Theta",
    "ThetaModule",
    "HParams",
    "InferenceModelConfig",
    "InferenceModel",
]


class UnpackedStruct(ABC):
    @abstractmethod
    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        ...


UnpackedStructT = TypeVar("UnpackedStructT", bound=UnpackedStruct)


class InferenceTensor(ABC):
    """Provides access to a tensor in the model used for inference.

    InferenceTensors have a richer structure than "normal" training tensors
    since they often involve a degree of layout on top of the raw data tensor.
    """

    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape


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


class QuantizedTensor(InferenceTensor, Generic[UnpackedStructT]):
    """An inference tensor that is quantized/packed."""

    def __init__(
        self,
        name: str,
        shape: list[int],
        *,
        struct_type: Type[UnpackedStruct],
    ):
        super().__init__(name, shape)
        self.struct_type = struct_type

    @abstractmethod
    def unpack(self) -> UnpackedStructT:
        ...

    @abstractproperty
    def raw(self) -> torch.Tensor:
        ...


class Theta:
    """Subset of parameter tensors used for inference."""

    def __init__(self, tensors: dict[str, InferenceTensor]):
        assert all(
            isinstance(t, InferenceTensor) for t in tensors.values()
        ), "Must only contain InferenceTensors"

        self._tensors = _flat_to_nested_dict(tensors)

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
            raise KeyError(f"Unknown parameter {name_path}")
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
        return Theta(current_ts)

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


class HParams(dict[str, Any]):
    """Model level hyper-parameters.

    HParams are dict objects that can also mix in specific vocabularies for
    various typed accessors.
    """

    ...


class ThetaModule(nn.Module):
    """An nn module which operates on parameters contained in a theta object."""

    def __init__(self, theta: Theta):
        super().__init__()
        self.theta = theta


@dataclass
class InferenceModelConfig:
    """Top level configuration for a model.

    This consists of:

    * Model level hyper-parameters (HParams).
    * Root theta with materialized parameters (Theta).
    """

    hp: HParams
    root_theta: Theta


class InferenceModel(ThetaModule):
    """Top-level inference model."""

    def __init__(self, config: InferenceModelConfig):
        super().__init__(config.root_theta)
        self.hp = config.hp
