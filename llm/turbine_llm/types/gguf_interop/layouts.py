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
    SuperBlockOffsetScaled_4_6_Layout,
)

from ..layout_utils import (
    linearize_interleaved_i4_block,
)

__all__ = [
    "Q4_1",
    "Q4_K",
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


class Q4_K(QuantizedTensor[SuperBlockOffsetScaled_4_6_Layout]):
    """Implements the Q4_K quantization scheme.

    ```
    #define QK_K 256
    #define K_SCALE_SIZE 12
    typedef struct {
        union {
            struct {
                ggml_half d;    // super-block scale for quantized scales
                ggml_half dmin; // super-block scale for quantized mins
            } GGML_COMMON_AGGR;
            ggml_half2 dm;
        };
        uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
        uint8_t qs[QK_K/2];           // 4--bit quants
    } block_q4_K;
    ```

    This uses the same 6bit scales and mins packing scheme and super-block
    structure as some other "K" quantizations. Because the format was the subject
    of a fair amount of creativity, we have to go to some considerable lengths
    to planarize it, with the novel piece that we unpack the inner block
    scales and mins to 4 arrays with POT bit depths.

    * 8 * i4 : uint8 ms_low[4]
    * 8 * i2 : uint8 ms_hi[2]
    * 8 * i4 : uint8 ds_low[4]
    * 8 * i2 : uint8 ds_hi[2]

    This gives us the characteristic of linear addressing on the components,
    which the compiler can do more with than a heavily interleaved format.
    """

    def __init__(self, *, name: str, raw: torch.Tensor, shape: list[int]):
        super().__init__(
            name, shape=shape, layout_type=SuperBlockOffsetScaled_4_6_Layout
        )
        self.raw = raw

    def unpack(self) -> SuperBlockOffsetScaled_4_6_Layout:
        # Blocks are 72 i16s, so start there.
        # [0] f16: d
        # [1] f16: dmin
        # [2:8] 12 * i8: 6 * i16: scales, mins
        # [8:72] 128 * i8: 64 * i16: qs
        linear_blocks = self.raw.view(torch.int16).reshape(-1, 72)
        # Reblock to the result shape, excluding the final dimension, which is
        # expanded.
        block_shape = self.shape[0:-1] + [-1, 72]
        blocks = linear_blocks.reshape(block_shape)
        d = blocks[..., 0:1].view(torch.float16)
        dmin = blocks[..., 1:2].view(torch.float16)
        raw_sb_scale_mins = blocks[..., 2:8].view(torch.uint8)
        scales_high, scales_low, mins_high, mins_low = _unpack_gguf_i6_scale_mins(
            raw_sb_scale_mins
        )

        # TODO: qs are not swizzled correctly.
        qs_raw = blocks[..., 8:].view(torch.uint8)
        # print(f"QS_RAW: {qs_raw.shape}")
        qs_blocked = qs_raw.unflatten(dim=-1, sizes=(4, -1))
        # print(f"QS_BLOCKED: {qs_blocked.shape}")
        # qs = qs_blocked
        # De-interleave at the 2 sub-block granularity (64 qs).
        qs = linearize_interleaved_i4_block(qs_blocked)
        # Then reshape it back to the view of 8 32 sample blocks.
        qs = qs.flatten(-2).unflatten(dim=-1, sizes=(8, -1))
        # print(f"QS: {qs.shape}")
        return SuperBlockOffsetScaled_4_6_Layout(
            self.shape,
            d=d,
            dmin=dmin,
            sb_scales_high=scales_high,
            sb_scales_low=scales_low,
            sb_mins_high=mins_high,
            sb_mins_low=mins_low,
            qs=qs,
        )

    @property
    def globals(self):
        return {self.name: self.raw}

    def __repr__(self):
        return f"Q4_K({self.name}, {self.shape})"


def _unpack_gguf_i6_scale_mins(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # GGML bit-packs 16 6 bit scales/mins into 12 bytes with a fairly bespoke
    # layout. See here for a visualization:
    # https://docs.google.com/spreadsheets/d/1XbwCZRTQiXaEHB3PVM1FgMcLlplzmUB9JD5olVZj9ks/edit?usp=sharing
    # We unpack it into four planar tensors with the low 4 bits and high 2 bits
    # of each separate. We refer to scales as `d` and mins as `m` for brevity.
    assert raw.dtype == torch.uint8
    assert raw.size(-1) == 12

    # d_low
    d_0_1_low = (raw[..., 0] & 0xF) | ((raw[..., 1] & 0xF) << 4)
    d_2_3_low = (raw[..., 2] & 0xF) | ((raw[..., 3] & 0xF) << 4)
    d_4_5_low = (raw[..., 8] & 0xF) | ((raw[..., 9] & 0xF) << 4)
    d_6_7_low = (raw[..., 10] & 0xF) | ((raw[..., 11] & 0xF) << 4)
    d_low = torch.stack([d_0_1_low, d_2_3_low, d_4_5_low, d_6_7_low], dim=-1)

    # d_high
    d_0_3_high = (
        ((raw[..., 0] & 0x30) >> 4)
        | ((raw[..., 1] & 0x30) >> 2)
        | (raw[..., 2] & 0x30)
        | ((raw[..., 3] & 0x30) << 2)
    )
    d_4_7_high = (
        ((raw[..., 0] & 0xC0) >> 6)
        | ((raw[..., 1] & 0xC0) >> 4)
        | ((raw[..., 2] & 0xC0) >> 2)
        | (raw[..., 3] & 0xC0)
    )
    d_high = torch.stack([d_0_3_high, d_4_7_high], dim=-1)

    # m_low
    m_0_1_low = (raw[..., 4] & 0xF) | ((raw[..., 5] & 0xF) << 4)
    m_2_3_low = (raw[..., 6] & 0xF) | ((raw[..., 7] & 0xF) << 4)
    m_4_5_low = (raw[..., 8] >> 4) | (raw[..., 9] & 0xF0)
    m_6_7_low = (raw[..., 10] >> 4) | (raw[..., 11] & 0xF0)
    m_low = torch.stack([m_0_1_low, m_2_3_low, m_4_5_low, m_6_7_low], dim=-1)

    # m_high
    m_0_3_high = (
        ((raw[..., 4] & 0x30) >> 4)
        | ((raw[..., 5] & 0x30) >> 2)
        | (raw[..., 6] & 0x30)
        | ((raw[..., 7] & 0x30) << 2)
    )
    m_4_7_high = (
        ((raw[..., 4] & 0xC0) >> 6)
        | ((raw[..., 5] & 0xC0) >> 4)
        | ((raw[..., 6] & 0xC0) >> 2)
        | (raw[..., 7] & 0xC0)
    )
    m_high = torch.stack([m_0_3_high, m_4_7_high], dim=-1)

    return d_high, d_low, m_high, m_low


class Q5_K(QuantizedTensor[BlockScaledI4Layout]):
    """"""

    def __init__(self, *, name: str, raw: torch.Tensor, shape: list[int]):
        super().__init__(name, shape=shape, layout_type=BlockScaledI4Layout)
        self.raw = raw

    def unpack(self) -> BlockScaledI4Layout:
        raise NotImplementedError

    @property
    def globals(self):
        return {self.name: self.raw}

    def __repr__(self):
        return f"Q5_K({self.name}, {self.shape})"


class Q6_K(QuantizedTensor[BlockScaledI4Layout]):
    """"""

    def __init__(self, *, name: str, raw: torch.Tensor, shape: list[int]):
        super().__init__(name, shape=shape, layout_type=BlockScaledI4Layout)
        self.raw = raw

    def unpack(self) -> BlockScaledI4Layout:
        raise NotImplementedError

    @property
    def globals(self):
        return {self.name: self.raw}

    def __repr__(self):
        return f"Q6_K({self.name}, {self.shape})"


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
