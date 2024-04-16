# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from turbine_llm.types.layout_utils import *


class I4Shuffle(unittest.TestCase):
    def test_linearize_interleaved_i4_block(self):
        # Linearize.
        input_data = torch.tensor(
            [0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7], dtype=torch.uint8
        ).unsqueeze(0)
        linear = linearize_interleaved_i4_block(input_data)
        self.assertEqual(
            r"[['0x10', '0x32', '0x54', '0x76', '0x98', '0xba', '0xdc', '0xfe']]",
            repr(debug_map_tensor_as_hex_string(linear)),
        )

        # Invert back to interleaved.
        interleaved = interleave_linear_i4_block(linear)
        self.assertEqual(
            r"[['0x80', '0x91', '0xa2', '0xb3', '0xc4', '0xd5', '0xe6', '0xf7']]",
            repr(debug_map_tensor_as_hex_string(interleaved)),
        )

    def test_promote_i4_block_to_i8_unsigned(self):
        # Start with linear data.
        linear_i4_data = torch.tensor(
            [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8
        ).unsqueeze(0)
        r0 = promote_linear_i4_block_to_i8(linear_i4_data)
        self.assertEqual(r0.dtype, torch.uint8)
        torch.testing.assert_close(
            torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
                dtype=torch.uint8,
            ),
            r0,
        )

    def test_promote_i4_block_to_i8_signed(self):
        # Start with linear data.
        linear_i4_data = (
            torch.tensor(
                [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8
            )
            .unsqueeze(0)
            .view(torch.uint8)
        )
        r0 = promote_linear_i4_block_to_i8(linear_i4_data, signed=True)
        self.assertEqual(r0.dtype, torch.int8)
        torch.testing.assert_close(
            torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1]],
                dtype=torch.int8,
            ),
            r0,
        )


if __name__ == "__main__":
    unittest.main()
