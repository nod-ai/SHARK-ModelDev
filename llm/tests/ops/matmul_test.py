# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest

import torch

from turbine_llm import ops
from turbine_llm.types import layout_utils


class mmtfp_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test2DF32(self):
        a = torch.rand([128, 32], dtype=torch.float32)
        b = torch.rand([256, 32], dtype=torch.float32)
        result = ops.mmtfp(a, b)
        torch.testing.assert_close(result, torch.matmul(a, b.T))

    def test3DF32(self):
        a = torch.rand([4, 128, 32], dtype=torch.float32)
        b = torch.rand([256, 32], dtype=torch.float32)
        result = ops.mmtfp(a, b)
        torch.testing.assert_close(result, torch.matmul(a, b.T))


class mmt_block_scaled_q8_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def testF32BS32(self):
        a = torch.rand([4, 16, 3200], dtype=torch.float32)
        d = torch.rand([3200, 100, 1], dtype=torch.float16)
        qs = (torch.rand([3200, 100, 32], dtype=torch.float32) * 32.0).to(torch.int8)
        result = ops.mmt_block_scaled_q8(a, d, qs)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        b = (d.to(torch.float32) * qs.to(torch.float32)).flatten(1)
        torch.testing.assert_close(result, torch.matmul(a, b.T), atol=1e-1, rtol=1e-5)


class mmt_block_scaled_offset_q4_unsigned_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_basic(self):
        a = torch.rand([4, 16, 3200], dtype=torch.float32)
        d = torch.rand([3200, 100, 1], dtype=torch.float16)
        qs = (torch.rand([3200, 100, 16], dtype=torch.float32) * 32).to(torch.uint8)
        m = torch.rand([3200, 100, 1], dtype=torch.float16)
        result = ops.mmt_block_scaled_offset_q4_unsigned(a, d, qs, m)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        qs_i8 = layout_utils.promote_linear_i4_block_to_i8(qs)
        b = (d.to(torch.float32) * qs_i8.to(torch.float32) + m).flatten(1)
        torch.testing.assert_close(result, torch.matmul(a, b.T), atol=1e-1, rtol=1e-5)


class mmt_super_block_scaled_offset_q4_unsigned(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @unittest.skip(
        "compiler bad tile selection:"
        "https://github.com/openxla/iree/issues/17078#issuecomment-2062331207"
    )
    def test_basic(self):
        # n = 2560, k = 5120, sup = 20, sub = 8, bs = 32
        a = torch.rand([4, 16, 5120], dtype=torch.float32)
        d = torch.rand([2560, 20, 1], dtype=torch.float16)
        dmin = torch.rand([2560, 20, 1], dtype=torch.float16)
        sb_scales_hi = (torch.rand([2560, 20, 2], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        sb_scales_low = (torch.rand([2560, 20, 4], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        sb_mins_hi = (torch.rand([2560, 20, 2], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        sb_mins_low = (torch.rand([2560, 20, 4], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        qs = (torch.rand([2560, 20, 8, 16], dtype=torch.float32) * 127).to(torch.uint8)
        result = ops.mmt_super_block_scaled_offset_q4_unsigned(
            a, d, dmin, sb_scales_hi, sb_scales_low, sb_mins_hi, sb_mins_low, qs
        )
        # TODO: Validate numerics once enabled and crash bug fixed.


if __name__ == "__main__":
    unittest.main()
