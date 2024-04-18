# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

__all__ = [
    "debug_map_tensor_as_hex_string",
    "interleave_linear_i4_block",
    "linearize_interleaved_i4_block",
    "promote_linear_i2_block_to_i8",
    "promote_linear_i4_block_to_i8",
    "promote_linear_i6_block_to_i8",
]


def linearize_interleaved_i4_block(i8_data: torch.Tensor) -> torch.Tensor:
    """De-interleaves a tensor with an i4 block of data as its innermost dim.

    Given 4bit data of the form:
        0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7
    Converts to a linear form:
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe

    Such interleaved data often is a natural form for broadcasting direct from
    tensors of low and high nibbles to a larger bit-width, so it shows up a lot.
    The linearized version can be more useful for algorithms that are operating
    on a packed block directly or that prefer such register layouts.
    """
    i8_data = _view_uint8_tensor(i8_data)
    assert i8_data.dtype == torch.uint8, f"Expected uint8. Got {i8_data.dtype}"
    low_nibbles = i8_data & 0xF
    high_nibbles = i8_data >> 4
    low_even = low_nibbles[..., ::2]
    low_odd = low_nibbles[..., 1::2]
    high_even = high_nibbles[..., ::2]
    high_odd = high_nibbles[..., 1::2]
    t1 = (low_odd << 4) | low_even
    t2 = (high_odd << 4) | high_even
    linear = torch.cat([t1, t2], dim=-1)
    return linear


def interleave_linear_i4_block(i8_data: torch.Tensor) -> torch.Tensor:
    """Inverse of linearize_interleaved_i4_block."""
    i8_data = _view_uint8_tensor(i8_data)
    t1, t2 = torch.tensor_split(i8_data, 2, dim=-1)
    assert t1.size(-1) == t2.size(
        -1
    ), "interleave_linear_i4_block: must have even inner-most dim"
    low_even = t1 & 0xF
    low_odd = t1 >> 4
    high_even = t2 & 0xF
    high_odd = t2 >> 4
    i0 = (high_even << 4) | low_even
    i1 = (high_odd << 4) | low_odd
    i0 = i0.unsqueeze(-1)
    i1 = i1.unsqueeze(-1)
    stacked = torch.cat([i0, i1], dim=-1)
    interleaved = stacked.flatten(start_dim=-2)
    return interleaved


def promote_linear_i4_block_to_i8(
    linear_i4_data: torch.Tensor, *, signed: bool = False
) -> torch.Tensor:
    """Promote a linear i4 blocked tensor to i8."""
    linear_i4_data = _view_uint8_tensor(linear_i4_data)
    if signed:
        # For signed i4 quantities, we have to manipulate the values as
        # right shifts from the high order nibble in order for sign extension
        # to function.
        low = (linear_i4_data << 4).view(torch.int8) >> 4
        high = linear_i4_data.view(torch.int8) >> 4
    else:
        low = linear_i4_data & 0xF
        high = linear_i4_data >> 4

    low = low.unsqueeze(-1)
    high = high.unsqueeze(-1)
    stacked = torch.cat([low, high], dim=-1)
    flat = stacked.flatten(start_dim=-2)
    return flat


def promote_linear_i2_block_to_i8(linear_i2_data: torch.Tensor) -> torch.Tensor:
    """Promote a linear i4 blocked tensor to i8."""
    linear_i2_data = _view_uint8_tensor(linear_i2_data)
    assert linear_i2_data.dtype == torch.uint8, "NYI: Signed i2 promote to i8"
    d0 = linear_i2_data & 0x3
    d1 = (linear_i2_data >> 2) & 0x3
    d2 = (linear_i2_data >> 4) & 0x3
    d3 = (linear_i2_data >> 6) & 0x3
    stacked = torch.cat(
        [d0.unsqueeze(-1), d1.unsqueeze(-1), d2.unsqueeze(-1), d3.unsqueeze(-1)], dim=-1
    )
    flat = stacked.flatten(start_dim=-2)
    return flat


def promote_linear_i6_block_to_i8(
    i6_data_high: torch.Tensor, i6_data_low: torch.Tensor
) -> torch.Tensor:
    """Combines a 4 bit and 2 bit tensor into i8 values."""
    i4_data_low = promote_linear_i4_block_to_i8(i6_data_low)
    i2_data_high = promote_linear_i2_block_to_i8(i6_data_high)
    assert (
        i4_data_low.shape == i2_data_high.shape
    ), f"i4 low/high tensors should have the same shape ({i4_data_low.shape} vs {i2_data_high.shape})"
    return i4_data_low | (i2_data_high << 4)


def debug_map_tensor_as_hex_string(data: torch.Tensor) -> list:
    """Debug helper to print contents of a tensor mapped via hex().

    Returns a list with the same structure as the tensor but with all elements
    replaced with a hexadecimal string representation. Useful for debugging
    transformations on binary tensors.
    """

    def mapelt(x):
        if isinstance(x, list):
            return [mapelt(y) for y in x]
        return hex(x)

    return mapelt(data.tolist())


def _view_uint8_tensor(data: torch.Tensor) -> torch.Tensor:
    """Views an int8/uint8 tensor as uint8.

    Asserts if any other dtype.

    This helper is for performing raw bitwise manipulations on sub-byte values.
    If doing arithmetic bitwise, you will want to use a signed tensor and
    appropriate operations to manage sign extension.
    """
    dtype = data.dtype
    if dtype == torch.uint8:
        return data
    elif dtype == torch.int8:
        return data.view(torch.uint8)
    else:
        raise AssertionError(f"Expected tensor to by uint8 or int8. Got: {dtype}")
