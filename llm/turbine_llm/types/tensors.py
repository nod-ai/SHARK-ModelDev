# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, Union, Collection, TypeVar, Generic, Type

from abc import ABC, abstractmethod

import torch

__all__ = [
    "InferenceTensor",
    "PrimitiveTensor",
    "QuantizedTensor",
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
    def as_torch(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
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

    def as_torch(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is not None:
            return self._data.to(dtype)
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
