# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from ..tensors import (
    QuantizedTensor,
)

from ..layouts import (
    BlockScaledLayout,
    BlockScaledI4Layout,
)

from ..layout_utils import (
    linearize_interleaved_i4_block,
)

__all__ = [
    "Q4_1",
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


class Q4_1(QuantizedTensor[BlockScaledI4Layout]):
    """Support quantized tensors in the Q4_1 format.

    ```
    #define QK4_1 32
    typedef struct {
        ggml_fp16_t d;          // delta
        ggml_fp16_t m;          // min
        uint8_t qs[QK4_1 / 2];  // nibbles / quants
    } block_q4_1;
    ```

    This scheme has some quirks:

    * The `qs` are interleaved vs laid out linearly.
    * Nibbles are unsigned quantities.
    * `m` is pre-scaled by `delta`.
    """

    def __init__(self, *, name: str, raw: torch.Tensor, shape: list[int]):
        super().__init__(name, shape=shape, layout_type=BlockScaledI4Layout)
        self.raw = raw

    def unpack(self) -> BlockScaledI4Layout:
        # Blocks are 10 i16s, so start there.
        linear_blocks = self.raw.view(torch.int16).reshape(-1, 10)
        # Reblock to the result shape excluding the final dimension, which
        # is expanded.
        block_shape = self.shape[0:-1] + [-1, 10]
        blocks = linear_blocks.reshape(block_shape)
        d = blocks[..., 0:1].view(torch.float16)
        m = blocks[..., 1:2].view(torch.float16)
        qs_raw = blocks[..., 2:].view(torch.int8)
        # GGML packing of Q4 data is in the order:
        # [0, 16, 1, 17, 2, 18, ...]
        # We need to repack to the [0, 1, 2, ...] order, which we define as
        # the "correct" basis packing.
        qs = linearize_interleaved_i4_block(qs_raw)
        return BlockScaledI4Layout(self.shape, d, qs, m=m, signed=False)

    @property
    def globals(self):
        return {self.name: self.raw}

    def __repr__(self):
        return f"Q4_1({self.name}, {self.shape})"


# TODO: Bring the other variants over from ggml_structs.py.
