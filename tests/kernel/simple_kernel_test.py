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
    def testSimpleKernel(self):
        def add_vectors_kernel(x, y, out):
            out[:] = x + y

        example_x = torch.empty(5)
        example_y = torch.empty(5)
        example_out = torch.empty(5)

        add_vectors_kernel(example_x, example_y, example_out)
        print("OUT =", example_out)

        exp_f = dynamo.export(add_vectors_kernel, aten_graph=True, assume_static_by_default=True, same_signature=False)
        gm, guards = exp_f(example_x, example_y, example_out)
        #print(gm)
        gm.print_readable()



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
