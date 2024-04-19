# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest

import torch

from shark_turbine import aot
from turbine_llm import ops
from turbine_llm.types import layout_utils


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

    def testExportDynamicDims(self):
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

        class MyModule(torch.nn.Module):
            def forward(
                self,
                a,
                d,
                dmin,
                sb_scales_hi,
                sb_scales_low,
                sb_mins_hi,
                sb_mins_low,
                qs,
            ):
                return ops.mmt_super_block_scaled_offset_q4_unsigned(
                    a, d, dmin, sb_scales_hi, sb_scales_low, sb_mins_hi, sb_mins_low, qs
                )

        mod = MyModule()
        batch = torch.export.Dim("batch")
        m = torch.export.Dim("m")
        ep = torch.export.export(
            mod,
            args=(a, d, dmin, sb_scales_hi, sb_scales_low, sb_mins_hi, sb_mins_low, qs),
            dynamic_shapes={
                "a": {0: batch, 1: m},
                "d": {},
                "dmin": {},
                "sb_scales_hi": {},
                "sb_scales_low": {},
                "sb_mins_hi": {},
                "sb_mins_low": {},
                "qs": {},
            },
        )
        asm = str(aot.export(ep).mlir_module)
        self.assertIn(
            "@mmt_super_block_scaled_offset_q4_unsigned_3d_2560_5120_20_8_32_f32", asm
        )


if __name__ == "__main__":
    unittest.main()
