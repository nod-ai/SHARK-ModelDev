# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Union

import os

import numpy as np

from gguf import GGUFReader, GGUFValueType

from ...utils.logging import get_logger

from ..base import (
    HParams,
    InferenceModelConfig,
    InferenceTensor,
    Theta,
)

from . import layouts

__all__ = [
    "load_gguf_file",
]

logger = get_logger("gguf")


class GgufHParams(HParams):
    def __init__(self, reader: GGUFReader):
        super().__init__()
        self._tables: dict[str, Any] = {}

        # Extract hyper-parameters. Adapted from gguf-dump.py
        for field in reader.fields.values():
            if len(field.types) == 1:
                curr_type = field.types[0]
                if curr_type == GGUFValueType.STRING:
                    self[field.name] = str(bytes(field.parts[-1]), encoding="utf8")
                elif field.types[0] in reader.gguf_scalar_to_np:
                    self[field.name] = field.parts[-1][0]
            else:
                self._tables[field.name] = field.parts


_quantized_types = {
    "Q8_0": layouts.Q8_0,
}


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
        return layouts.GgufPrimitiveTensor(
            name=name, shape=logical_shape, type_name=type_name, data=data
        )

    quantized_type = _quantized_types.get(type_name)
    if quantized_type is not None:
        return quantized_type(name=name, data=data, shape=logical_shape)

    raise ValueError(f"Unsupported gguf tensor type: {type_name}")


def load_gguf_file(gguf_path: Union[str, os.PathLike]):
    reader = GGUFReader(gguf_path)
    logger.info(
        "Loading gguf file %s (%d fields, %d tensors)",
        gguf_path,
        len(reader.fields),
        len(reader.tensors),
    )
    hp = GgufHParams(reader)

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
    return InferenceModelConfig(hp=hp, root_theta=root_theta)
