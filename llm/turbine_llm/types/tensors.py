# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union, TypeVar, Generic, Type

from abc import ABC, abstractmethod

import torch

from shark_turbine.aot import ParameterArchiveBuilder

__all__ = [
    "register_quantized_layout",
    "InferenceTensor",
    "MetaDataValueType",
    "PrimitiveTensor",
    "QuantizedTensor",
    "QuantizedLayout",
]

# JSON encodable value types.
MetaDataValueType = Union[int, bool, float, str]


class QuantizedLayout(ABC):
    @abstractmethod
    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        ...

    @classmethod
    @abstractmethod
    def serialized_name(self) -> str:
        """Returns the globally unique serialization name for this layout."""
        ...

    @classmethod
    @abstractmethod
    def create(
        cls,
        shape: list[int],
        metadata: dict[str, MetaDataValueType],
        planes: dict[str, torch.Tensor],
    ):
        ...

    @property
    @abstractmethod
    def planes(self) -> dict[str, torch.Tensor]:
        """Returns the planes of this layout as concrete, named tensors.

        When transforming, the name will form a local suffix (i.e. ":name")
        for stored values by combining the global name with the ":" separator.
        """
        ...

    @property
    def metadata(self) -> dict[str, MetaDataValueType]:
        """Additional metadata needed to reconstruct a layout."""
        return {}

    def planarize(self) -> "QuantizedTensor":
        ...


QuantizedLayoutT = TypeVar("QuantizedLayoutT", bound=QuantizedLayout)


REGISTERED_LAYOUT_CLASSES: dict[str, Type[QuantizedLayoutT]] = {}


def register_quantized_layout(ty: Type[QuantizedLayoutT]) -> Type[QuantizedLayoutT]:
    """Class decorator which registers a serializable layout class."""
    name = ty.serialized_name()
    existing = REGISTERED_LAYOUT_CLASSES.get(name)
    assert (
        existing is None
    ), f"Duplicate QuantizedLayoutRegistration '{name}' ({ty} vs {existing})"
    REGISTERED_LAYOUT_CLASSES[name] = ty
    return ty


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

    @abstractmethod
    def add_to_archive(self, builder: ParameterArchiveBuilder):
        """Adds this tensor to the global archive."""
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

    def add_to_archive(self, builder: ParameterArchiveBuilder):
        """Adds this tensor to the global archive."""
        builder.add_tensor(self.name, self._data, metadata=repr(self))

    def __repr__(self):
        return f"PrimitiveTensor({self.name}, {self.shape}, {self._data.dtype})"


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

    def add_to_archive(self, builder: ParameterArchiveBuilder):
        """Adds this tensor to the global archive."""
        root_name = self.name
        layout = self.unpack()
        for suffix, plane in layout.planes.items():
            irpa_name = f"{root_name}:{suffix}"
            builder.add_tensor(irpa_name, plane, metadata=repr(self))

    @abstractmethod
    def unpack(self) -> QuantizedLayoutT:
        ...


class PlanarQuantizedTensor(QuantizedTensor):
    """Generic planar tensor backed by an instantiated QuantizedLayout.

    This is used for materialized, unpacked layouts (i.e. no unpacking
    will be done).
    """

    def __init__(self, name: str, shape: list[int], layout: QuantizedLayout):
        super().__init__(name, shape, layout_type=type(layout))
        self.layout = layout

    def unpack(self) -> QuantizedLayout:
        return self.layout

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        global_name = self.name
        planes = self.layout.planes
        return {f"{global_name}︴{k}": v for k, v in planes.items()}

    def __repr__(self):
        return (
            f"PlanarQuantized({self.name}, {self.shape}, planes={self.globals.keys()})"
        )
