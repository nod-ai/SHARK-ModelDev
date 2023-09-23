# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import torch._dynamo as dynamo

import shark_turbine.kernel as tk


class Test(unittest.TestCase):
    def testIota(self):
        @tk.block_kernel(eager=True)
        def iota_kernel(out: tk.KernelBuffer):
            i = tk.program_id(0)
            out[i] = i

        gridded = iota_kernel(grid=(8,))
        out = torch.empty(8, dtype=torch.int32)
        gridded(out)
        print(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
