# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, Union, TypeVar, Generic, Type

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    def serialized_name(cls) -> str:
        """Returns the globally unique serialization name for this layout."""
        ...

    @classmethod
    @abstractmethod
    def create(
        cls,
        shape: list[int],
        metadata: Optional[dict[str, MetaDataValueType]],
        planes: dict[str, torch.Tensor],
    ) -> "QuantizedLayout":
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
    def metadata(self) -> Optional[dict[str, MetaDataValueType]]:
        """Additional metadata needed to reconstruct a layout."""
        return None


QuantizedLayoutT = TypeVar("QuantizedLayoutT", bound=QuantizedLayout)


REGISTERED_LAYOUT_CLASSES: dict[str, Type[QuantizedLayout]] = {}


def register_quantized_layout(ty: Type[QuantizedLayoutT]) -> Type[QuantizedLayoutT]:
    """Class decorator which registers a serializable layout class."""
    name = ty.serialized_name()
    existing = REGISTERED_LAYOUT_CLASSES.get(name)
    assert (
        existing is None
    ), f"Duplicate QuantizedLayoutRegistration '{name}' ({ty} vs {existing})"
    REGISTERED_LAYOUT_CLASSES[name] = ty
    return ty


@dataclass
class InferenceTensorMetadata:
    # Registered name of an InferenceTensor subclass.
    type_name: str
    # Mapping of constituent local names to parameter archive global names
    # of individual tensors that make up this InferenceTensor.
    raw_tensors: dict[str, str]
    # Additional properties needed to restore the instance. Must be JSON
    # legal types. Will be added to the root JSON dictionary.
    extra_properties: Optional[dict[str, Any]] = None

    def create_instance(self) -> "InferenceTensor":
        try:
            clazz = REGISTERED_INFERENCE_TENSOR_CLASSES[self.type_name]
        except KeyError as e:
            raise IOError(
                f"Unable to create instance of unregistered type {self.type_name}"
            ) from e
        assert issubclass(clazz, InferenceTensor)

    def to_json(self) -> dict:
        d = {
            "type_name": self.type_name,
            "raw_tensors": self.raw_tensors,
        }
        if self.extra_properties is not None:
            d.update(self.extra_properties)
        return d

    def from_json(obj: dict) -> "InferenceTensorMetadata":
        extra_properties = dict(obj)
        try:
            type_name = extra_properties["type_name"]
            assert isinstance(type_name, str)
            del extra_properties["type_name"]
            raw_tensors = extra_properties["raw_tensors"]
            assert isinstance(raw_tensors, dict)
            del extra_properties["raw_tensors"]
        except Exception as e:
            raise IOError(f"Error decoding InferenceTensorMetadata object") from e

        # Validate.
        for k, v in raw_tensors.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise IOError(
                    f"Bad format for InferenceTensorMetadata.raw_tensors ({type(k)}, {type(v)})"
                )

        return InferenceTensorMetadata(
            type_name=type_name,
            raw_tensors=raw_tensors,
            extra_properties=extra_properties,
        )


class InferenceTensor(ABC):
    """Provides access to a tensor in the model used for inference.

    InferenceTensors have a richer structure than "normal" training tensors
    since they often involve a degree of layout on top of the raw data tensor.
    """

    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        raise NotImplementedError(
            f"InferenceTensor {cls} cannot be deserialized "
            f"because it does not implement create()"
        )

    @classmethod
    def serialized_name(cls) -> str:
        """Returns the globally unique serialization name for this type."""
        raise NotImplementedError(
            f"InferenceTensor {cls} cannot be directly "
            f"serialized (does not implement serialized_name())"
        )

    @property
    @abstractmethod
    def globals(self) -> dict[str, torch.Tensor]:
        """Returns a mapping of global name to root tensor.

        The primary accessors on an InferenceTensor access the root tensors in
        the global set, all of which in a root Theta must have unique names.
        """
        ...

    @abstractmethod
    def add_to_archive(
        self, builder: ParameterArchiveBuilder
    ) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        ...


REGISTERED_INFERENCE_TENSOR_CLASSES: dict[str, Type[InferenceTensor]] = {}


def register_inference_tensor(ty: Type[QuantizedLayoutT]) -> Type[QuantizedLayoutT]:
    """Class decorator which registers a serializable InferenceTensor class.

    This should only be used to decorate concrete implementations that need to
    be loaded by name.
    """
    name = ty.serialized_name()
    existing = REGISTERED_INFERENCE_TENSOR_CLASSES.get(name)
    assert (
        existing is None
    ), f"Duplicate InferenceTensor registration '{name}' ({ty} vs {existing})"
    REGISTERED_INFERENCE_TENSOR_CLASSES[name] = ty
    return ty


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


@register_inference_tensor
class DefaultPrimitiveTensor(PrimitiveTensor):
    """Concrete implementation of a PrimitiveTensor based on a single tensor."""

    def __init__(self, name: str, data: torch.Tensor):
        super().__init__(name, list(data.shape))
        self._data = data

    @classmethod
    def serialized_name(cls) -> str:
        return "PrimitiveTensor"

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        try:
            data = raw_tensors[""]
        except KeyError as e:
            raise IOError(f"Missing component tensor") from e
        return cls(name, data)

    def as_torch(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is not None:
            return self._data.to(dtype)
        return self._data

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        return {
            self.name: self._data,
        }

    def add_to_archive(
        self, builder: ParameterArchiveBuilder
    ) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        builder.add_tensor(self.name, self._data)
        return InferenceTensorMetadata(self.serialized_name(), {"": self.name})

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

    @abstractmethod
    def unpack(self) -> QuantizedLayoutT:
        ...

    def to_planar(self) -> "PlanarQuantizedTensor":
        """Converts this QuantizedTensor to a generic planar form.

        This is done for serialization and to materialize unpacking.
        If a subclass cannot be converted to planar form generically like this,
        it should override this method to implement properly or raise
        NotImplementedError.
        """
        return PlanarQuantizedTensor(self.name, self.shape, self.unpack())

    def add_to_archive(
        self, builder: ParameterArchiveBuilder
    ) -> InferenceTensorMetadata:
        """By default all QuantizedTensors serialize as a generic PlanarQuantizedTensor.

        If this is not desirable, subclasses should override.
        """
        return self.to_planar().add_to_archive(builder)


@register_inference_tensor
class PlanarQuantizedTensor(QuantizedTensor):
    """Generic planar tensor backed by an instantiated QuantizedLayout.

    This is used for materialized, unpacked layouts (i.e. no unpacking
    will be done).
    """

    def __init__(self, name: str, shape: list[int], layout: QuantizedLayout):
        super().__init__(name, shape, layout_type=type(layout))
        self.layout = layout

    def to_planar(self) -> "PlanarQuantizedTensor":
        # Already planar.
        return self

    @classmethod
    def serialized_name(cls) -> str:
        return "PlanarQuantizedTensor"

    def unpack(self) -> QuantizedLayout:
        return self.layout

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        global_name = self.name
        planes = self.layout.planes
        return {f"{global_name}:{k}": v for k, v in planes.items()}

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        try:
            shape = extra_properties["shape"]
            layout_type_name = extra_properties["layout_type"]
            layout_metadata = extra_properties.get("layout_metadata")
        except KeyError as e:
            raise IOError(f"Missing PlanarQuantizedTensor deserialization prop") from e

        shape = [int(d) for d in shape]
        try:
            layout_clazz = REGISTERED_LAYOUT_CLASSES[layout_type_name]
        except KeyError:
            raise IOError(
                f"Cannot deserialize PlanarQuantizedTensor because of unregistered layout "
                f"{layout_type_name}"
            )

        layout = layout_clazz.create(shape, layout_metadata, raw_tensors)
        return PlanarQuantizedTensor(name, shape, layout)

    def add_to_archive(
        self, builder: ParameterArchiveBuilder
    ) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        root_name = self.name
        layout = self.unpack()
        name_map: dict[str, str] = {}
        for suffix, plane in layout.planes.items():
            irpa_name = f"{root_name}:{suffix}"
            builder.add_tensor(irpa_name, plane)
            name_map[suffix] = irpa_name
        extra_properties = {
            "shape": [int(d) for d in self.shape],
            "layout_type": self.layout.serialized_name(),
        }
        layout_metadata = self.layout.metadata
        if layout_metadata is not None:
            extra_properties["layout_metadata"] = layout_metadata
        return InferenceTensorMetadata(
            PlanarQuantizedTensor.serialized_name(),
            name_map,
            extra_properties=extra_properties,
        )

    def __repr__(self):
        return (
            f"PlanarQuantized({self.name}, {self.shape}, planes={self.globals.keys()})"
        )
