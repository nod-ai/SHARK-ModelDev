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


class MmtFPTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
