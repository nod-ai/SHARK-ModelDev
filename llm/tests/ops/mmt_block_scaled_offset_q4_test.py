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

    def testExportDynamicDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, d, qs, m):
                return ops.mmt_block_scaled_offset_q4_unsigned(a, d, qs, m)

        mod = MyModule()
        batch = torch.export.Dim("batch")
        m = torch.export.Dim("m")
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 16, 3200], dtype=torch.float32),
                torch.rand([3200, 100, 1], dtype=torch.float16),
                (torch.rand([3200, 100, 16], dtype=torch.float32) * 32).to(torch.uint8),
                torch.rand([3200, 100, 1], dtype=torch.float16),
            ),
            dynamic_shapes={
                "a": {0: batch, 1: m},
                "d": {},
                "qs": {},
                "m": {},
            },
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn(
            "@turbine_llm_mmt_block_scaled_offset_q4_unsigned_3d_3200_3200_32_f32", asm
        )

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, d, qs, m):
                return ops.mmt_block_scaled_offset_q4_unsigned(a, d, qs, m)

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 16, 3200], dtype=torch.float32),
                torch.rand([3200, 100, 1], dtype=torch.float16),
                (torch.rand([3200, 100, 16], dtype=torch.float32) * 32).to(torch.uint8),
                torch.rand([3200, 100, 1], dtype=torch.float16),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn(
            "@turbine_llm_mmt_block_scaled_offset_q4_unsigned_3d_3200_3200_32_f32", asm
        )


if __name__ == "__main__":
    unittest.main()
