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


class mmtfp_test(unittest.TestCase):
    def test2DF32(self):
        result = ops.mmtfp(
            torch.rand([128, 32], dtype=torch.float32),
            torch.rand([256, 32], dtype=torch.float32),
        )
        print(result)
        # TODO: DO NOT SUBMIT: Add numerical test.

    def test3DF32(self):
        result = ops.mmtfp(
            torch.rand([4, 128, 32], dtype=torch.float32),
            torch.rand([256, 32], dtype=torch.float32),
        )
        print(result)
        # TODO: DO NOT SUBMIT: Add numerical test.


class mmt_block_scaled_q8_test(unittest.TestCase):
    def testF32BS32(self):
        a = torch.rand([4, 16, 3200], dtype=torch.float32)
        d = torch.rand([3200, 100, 1], dtype=torch.float16)
        qs = (torch.rand([3200, 100, 32], dtype=torch.float32) * 127.0).to(torch.int8)
        result = ops.mmt_block_scaled_q8(a, d, qs)
        print(result)


if __name__ == "__main__":
    unittest.main()
