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

    def testExportDynamicDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b, qs):
                return ops.mmt_block_scaled_q8(a, b, qs)

        mod = MyModule()
        batch = torch.export.Dim("batch")
        m = torch.export.Dim("m")
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 16, 3200], dtype=torch.float32),
                torch.rand([3200, 100, 1], dtype=torch.float16),
                (torch.rand([3200, 100, 32], dtype=torch.float32) * 32.0).to(
                    torch.int8
                ),
            ),
            dynamic_shapes={
                "a": {0: batch, 1: m},
                "b": {},
                "qs": {},
            },
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@turbine_llm_mmt_block_scaled_q8_3d_3200_3200_32_f32", asm)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b, qs):
                return ops.mmt_block_scaled_q8(a, b, qs)

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 16, 3200], dtype=torch.float32),
                torch.rand([3200, 100, 1], dtype=torch.float16),
                (torch.rand([3200, 100, 32], dtype=torch.float32) * 32.0).to(
                    torch.int8
                ),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@turbine_llm_mmt_block_scaled_q8_3d_3200_3200_32_f32", asm)


if __name__ == "__main__":
    unittest.main()
