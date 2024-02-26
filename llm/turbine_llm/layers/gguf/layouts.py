# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
import torch

from ..base import (
    PrimitiveTensor,
    QuantizedTensor,
    UnpackedStruct,
)

__all__ = [
    "GgufPrimitiveTensor",
    "Q8_0",
    "Q8_0Struct",
]


class GgufPrimitiveTensor(PrimitiveTensor):
    def __init__(self, name: str, shape: list[int], type_name: str, data: np.memmap):
        super().__init__(name, shape)
        self._type_name = type_name
        self._data = data

    def as_torch(self) -> torch.Tensor:
        return torch.Tensor(self._data).reshape(self.shape)

    def __repr__(self):
        return (
            f"GgufPrimitiveTensor({self.name}, {self.shape}, "
            f"dtype='{self._type_name}') = array({self._data.shape}, "
            f"dtype={self._data.dtype})"
        )


@dataclass
class Q8_0Struct(UnpackedStruct):
    shape: list[int]
    blocks: torch.Tensor
    d: torch.Tensor
    qs: torch.Tensor

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype).reshape(self.shape)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        qs = self.qs
        if dtype:
            d = d.to(dtype)
        else:
            dtype = d.dtype
        scaled = d * qs.to(dtype)
        return scaled

    def __repr__(self):
        return (
            f"Q8_0(d[{self.d.shape}, dtype={self.d.dtype}], "
            f"qs[{self.qs.shape}, dtype={self.qs.dtype}])"
        )


class Q8_0(QuantizedTensor[Q8_0Struct]):
    """
    ```
    #define QK8_0 32
    typedef struct {
        ggml_fp16_t d;         // delta
        int8_t  qs[QK8_0];     // quants
    } block_q8_0;
    ```
    Dequantize Q8_0:
    https://github.com/ggerganov/llama.cpp/blob/f026f8120f97090d34a52b3dc023c82e0ede3f7d/ggml-opencl.cpp#L172-L180
    """

    def __init__(self, *, name: str, data: np.memmap, shape: list[int]):
        super().__init__(name, shape, struct_type=Q8_0Struct)
        assert data.dtype == np.uint8
        self._data = data

    @property
    def raw(self) -> torch.Tensor:
        return torch.tensor(self._data)

    def unpack(self) -> Q8_0Struct:
        # Blocks are 17 i16s, so start there.
        linear_blocks = self.raw.view(torch.int16).reshape(-1, 17)
        # Reblock to the result shape excluding the final dimension, which
        # is expanded.
        block_shape = self.shape[0:-1] + [-1, 17]
        blocks = linear_blocks.reshape(block_shape)
        d = blocks[..., 0:1].view(torch.float16)
        qs = blocks[..., 1:].view(torch.int8)
        return Q8_0Struct(self.shape, blocks, d, qs)

    def __repr__(self):
        return f"Q8_0({self.name}, {self.shape})"


# TODO: Bring the other variants over from ggml_structs.py.
