# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional, Union, Collection

import json
from pathlib import Path
from types import NotImplementedType
from dataclasses import dataclass
import warnings

import torch
import torch.nn.functional as F

from shark_turbine.aot import (
    ParameterArchive,
    ParameterArchiveEntry,
    ParameterArchiveBuilder,
)

from .tensors import (
    InferenceTensor,
    PrimitiveTensor,
    QuantizedTensor,
    InferenceTensorMetadata,
    REGISTERED_INFERENCE_TENSOR_CLASSES,
)

__all__ = [
    "BaseInferenceOps",
    "Dataset",
    "Theta",
]

IOReportCallback = Callable[[str], None]


################################################################################
# Theta object
# A theta object represents a hierarchical pack of parameters. All parameters
# are InferenceTensor objects, meaning that they can either be raw PyTorch
# tensors or composite/packed QuantizedTensors.
#
# As in classic implementations, we separate the theta parameter pack from the
# model code because there are many interesting transformations that can be
# done on it in isolation.
#
# The theta object also carries with it side-car policy objects for making use
# of the contained tensors. Key among these is the concrete InferenceOps
# implementation for operating on the InferenceTensors.
################################################################################


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
        inference_tensor_metas: dict[str, InferenceTensorMetadata],
        *,
        io_report_callback: Optional[IOReportCallback] = None,
    ):
        """Adds tensors to the given archive builder."""
        for inference_tensor in self.flatten().values():
            if io_report_callback:
                io_report_callback(f"Add {inference_tensor}")
            name = inference_tensor.name
            if name in inference_tensor_metas:
                warnings.warn(f"Duplicate inference tensor added to archive: {name}")
            meta = inference_tensor.add_to_archive(irpa)
            inference_tensor_metas[name] = meta


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


################################################################################
# Dataset objects
#
# A dataset object combines a root theta and a dictionary of properties
# defining the computation characteristics that the parameters were trained for
# (i.e. hyperparams).
#
# Note that model implementation parameters are represented elsewhere (i.e. for
# things that involve selecting an implementation that meets certain
# characteristics).
################################################################################

PropertyValueType = Union[
    int, float, bool, list["PropertyValueType"], dict[str, "PropertyValueType"]
]


@dataclass
class Dataset:
    """Top level configuration for a model.

    This consists of:

    * Dataset level hyper-parameters (properties).
    * Root theta with materialized parameters (Theta).
    """

    properties: dict[str, PropertyValueType]
    root_theta: Theta

    def save(
        self,
        path: Union[str, Path],
        *,
        io_report_callback: Optional[IOReportCallback] = None,
    ):
        """Saves a parameter archive consisting of properties and theta.

        By default, all quantized tensors in theta which do not have a custom
        packed serialization are converted to a generic planar form.

        Sufficient metadata is stored such that `load()` can reconstitute the
        Dataset.
        """
        _dataset_save_helper(self, path, io_report_callback=io_report_callback)

    @staticmethod
    def load(path: Union[str, Path]) -> "Dataset":
        """Loads a dataset from a parameter archive constructed with save."""
        return _dataset_load_helper(path)


################################################################################
# Inference ops
# Key operations on InferenceTensors are represented here and our models and
# layers use them when appropriate. The default implementation defines them
# in terms of bespoke torch ops. Subclasses can be swapped that allow dispatch
# to more specialized implementations, which can be selected based on the
# characteristics of the tensors being operated on.
# TODO: This doesn't really belong here. Find it a better home.
################################################################################


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


################################################################################
# Dataset I/O helpers
################################################################################


@dataclass
class DatasetMetadata:
    """Container for serialization state of a dataset.

    When saved to an IRPA file, it will be saved with multiple keys:

    * properties: __SHARK_DATASET__
    * inference_tensors: __SHARK_INFERENCE_TENSORS__
    """

    properties: dict
    inference_tensors: dict[str, InferenceTensorMetadata]

    def save(
        self,
        builder: ParameterArchiveBuilder,
        *,
        io_report_callback: Optional[IOReportCallback] = None,
    ):
        properties_object = self.properties
        properties_object["SHARK_DATASET_VERSION"] = 1
        inference_tensors_object = {
            k: v.to_json() for k, v in self.inference_tensors.items()
        }

        # __SHARK_DATASET__ properties blob.
        try:
            properties_json_blob = json.dumps(properties_object, indent=2)
        except TypeError as e:
            raise TypeError(
                f"Illegal dataset properties object: {properties_object}"
            ) from e
        if io_report_callback:
            import textwrap

            io_report_callback(
                f"Add __SHARK_DATASET__:\n{textwrap.indent(properties_json_blob, '    ')}"
            )
        builder.add_blob("__SHARK_DATASET__", properties_json_blob.encode())

        # __SHARK_INFERENCE_TENSORS__ blob.
        try:
            inference_tensors_blob = json.dumps(inference_tensors_object, indent=2)
        except TypeError as e:
            raise TypeError(
                f"Illegal inference tensor object: {inference_tensors_object}"
            ) from e
        if io_report_callback:
            import textwrap

            io_report_callback(
                f"Add __SHARK_INFERENCE_TENSORS__:\n{textwrap.indent(inference_tensors_blob, '    ')}"
            )
        builder.add_blob("__SHARK_INFERENCE_TENSORS__", inference_tensors_blob.encode())

    @staticmethod
    def load(entries: dict[str, ParameterArchiveEntry]) -> "DatasetMetadata":
        # Load properties.
        try:
            properties_entry = entries["__SHARK_DATASET__"]
        except KeyError:
            raise IOError(
                f"Parameter archive does not contains __SHARK_DATASET__. Was it produced by this tool?"
            )
        properties_obj = json.loads(bytes(properties_entry.raw.file_view))
        assert isinstance(properties_obj, dict)

        # Load inference tensors.
        try:
            inference_tensors_entry = entries["__SHARK_INFERENCE_TENSORS__"]
        except KeyError:
            raise IOError(
                f"Parameter archive does not contains __SHARK_INFERENCE_TENSORS__. Was it produced by this tool?"
            )
        inference_tensors_obj = json.loads(bytes(inference_tensors_entry.raw.file_view))
        assert isinstance(inference_tensors_obj, dict)

        inference_tensors: dict[str, InferenceTensorMetadata] = {}
        for tensor_name, tensor_meta_obj in inference_tensors_obj.items():
            tensor_meta = InferenceTensorMetadata.from_json(tensor_meta_obj)
            # Map the raw_tensors dict to tensors from the archive.
            raw_tensors = {}
            for local_name, global_name in tensor_meta.raw_tensors.items():
                try:
                    raw_entry = entries[global_name]
                except KeyError as e:
                    raise IOError(f"InferenceTensor missing one of its tensor components") from e
                raw_tensor = raw_entry.as_tensor()
                raw_tensors[local_name] = raw_tensor

            # Instantiate the tensor.
            try:
                tensor_clazz = REGISTERED_INFERENCE_TENSOR_CLASSES[tensor_meta.type_name]
            except KeyError as e:
                raise IOError(f"Unregistered InferenceTensor deserialization type") from e
            inference_tensor = tensor_clazz.create(tensor_name, raw_tensors, tensor_meta.extra_properties)
            inference_tensors[tensor_name] = inference_tensor

        return DatasetMetadata(properties_obj, inference_tensors)


def _dataset_save_helper(
    dataset: Dataset,
    path: Union[str, Path],
    *,
    io_report_callback: Optional[IOReportCallback] = None,
):
    builder = ParameterArchiveBuilder()
    ds_meta = DatasetMetadata(dict(dataset.properties), {})
    # Add tensors.
    # TODO: We need some form of streaming save upstream and then use that.
    # For computed tensors, the memory overhead of this style is large
    # because intermediates are retained until the `builder.save()`.
    dataset.root_theta.add_tensors_to_archive(
        builder,
        ds_meta.inference_tensors,
        io_report_callback=io_report_callback,
    )

    # Serialize the metadata.
    ds_meta.save(builder, io_report_callback=io_report_callback)

    if io_report_callback:
        io_report_callback("Saving file")
    builder.save(path)


def _dataset_load_helper(
    path: Union[str, Path], *, file_type: Optional[str] = None
) -> Dataset:
    path = Path(path)
    suffixes = path.suffixes
    if file_type == "gguf" or suffixes == [".gguf"]:
        from . import gguf_interop

        return gguf_interop.load_file(path)
    elif file_type == "irpa" or suffixes == [".irpa"]:
        return _dataset_load_irpa(path)
    else:
        raise IOError(
            f"Unknown file type '{''.join(path.suffixes)} for loading a Dataset"
        )


def _dataset_load_irpa(path: Path) -> Dataset:
    archive = ParameterArchive(path)
    # Note that there may be duplicates. Last wins.
    entries = {k: v for k, v in archive.items()}
    meta = DatasetMetadata.load(entries)
    dataset = Dataset(meta.properties, Theta(meta.inference_tensors))
    return dataset
