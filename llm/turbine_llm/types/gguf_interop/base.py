# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, Union

import os

import numpy as np
import torch

from gguf import GGUFReader, GGUFValueType

from shark_turbine.aot import (
    ExternalTensorTrait,
)

from ...utils.logging import get_logger

from ..tensors import (
    DefaultPrimitiveTensor,
    InferenceTensor,
)

from ..theta import (
    Dataset,
    Theta,
)

from . import layouts

__all__ = [
    "load_file",
]

logger = get_logger("gguf")


def _load_properties(reader: GGUFReader) -> dict[str, Any]:
    # TODO: Figure out what to do with tables.
    tables: dict[str, Any] = {}
    properties: dict[str, Any] = {
        "schema": "GGUF",
        # "tables": tables,
    }

    # Extract hyper-parameters. Adapted from gguf-dump.py
    for field in reader.fields.values():
        if len(field.types) == 1:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                properties[field.name] = str(bytes(field.parts[-1]), encoding="utf8")
            elif field.types[0] in reader.gguf_scalar_to_np:
                properties[field.name] = field.parts[-1][0]
        else:
            tables[field.name] = field.parts
    return properties


_quantized_types = {
    "Q4_1": layouts.Q4_1,
    "Q8_0": layouts.Q8_0,
}


def _externalize_tensor(
    name: str, data: np.memmap, logical_shape: Optional[list[int]] = None
) -> torch.Tensor:
    # Important: The annotation tag must be set on the actual leaf tensor
    # which is stored in the root theta. This means that any shaping or
    # data type massaging has to happen *before* annotating.
    data_tensor = torch.tensor(data)
    if logical_shape is not None:
        data_tensor = data_tensor.reshape(logical_shape)
    ExternalTensorTrait(external_name=name, external_scope="").set(data_tensor)
    return data_tensor


def _wrap_tensor(
    name: str, logical_shape: list[int], type_name: str, data: np.memmap
) -> InferenceTensor:
    # Gguf internally optimizes for constant RHS and stores all weights
    # transposed. So we reverse the reported logical shape. Most operations
    # are then logically done with a transposed RHS.
    # TODO: This needs some more investigation to ensure that it is in fact
    # always true.
    logical_shape = list(reversed(logical_shape))
    if type_name in ["F16", "F32", "F64"]:
        return DefaultPrimitiveTensor(
            name, _externalize_tensor(name, data, logical_shape)
        )

    quantized_type = _quantized_types.get(type_name)
    if quantized_type is not None:
        return quantized_type(
            name=name,
            raw=_externalize_tensor(name, data),
            shape=logical_shape,
        )

    raise ValueError(f"Unsupported gguf tensor type: {type_name}")


def load_file(gguf_path: Union[str, os.PathLike]) -> Dataset:
    reader = GGUFReader(gguf_path)
    logger.info(
        "Loading gguf file %s (%d fields, %d tensors)",
        gguf_path,
        len(reader.fields),
        len(reader.tensors),
    )
    properties = _load_properties(reader)

    # Extract tensors.
    tensors: dict[str, InferenceTensor] = {}
    for tensor in reader.tensors:
        gguf_tensor = _wrap_tensor(
            name=tensor.name,
            logical_shape=list(tensor.shape),
            type_name=tensor.tensor_type.name,
            data=tensor.data,  # type: ignore
        )
        tensors[tensor.name] = gguf_tensor
    root_theta = Theta(tensors)
    return Dataset(properties=properties, root_theta=root_theta)
