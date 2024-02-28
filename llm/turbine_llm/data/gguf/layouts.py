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
    QuantizedTensor,
)

from ..layouts import (
    BlockScaledLayout,
)

__all__ = [
    "Q8_0",
]


class Q8_0(QuantizedTensor[BlockScaledLayout]):
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

    def __init__(self, *, name: str, raw: torch.Tensor, shape: list[int]):
        super().__init__(name, shape=shape, layout_type=BlockScaledLayout)
        assert raw.dtype == torch.uint8
        self.raw = raw

    def unpack(self) -> BlockScaledLayout:
        # Blocks are 17 i16s, so start there.
        linear_blocks = self.raw.view(torch.int16).reshape(-1, 17)
        # Reblock to the result shape excluding the final dimension, which
        # is expanded.
        block_shape = self.shape[0:-1] + [-1, 17]
        blocks = linear_blocks.reshape(block_shape)
        d = blocks[..., 0:1].view(torch.float16)
        qs = blocks[..., 1:].view(torch.int8)
        return BlockScaledLayout(self.shape, d, qs)

    @property
    def globals(self):
        return {self.name: self.raw}

    def __repr__(self):
        return f"Q8_0({self.name}, {self.shape})"


# TODO: Bring the other variants over from ggml_structs.py.
