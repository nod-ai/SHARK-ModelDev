# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import unittest

import torch

from testutils import *



class ImportTests(unittest.TestCase):
    def _create_model(self, bias):
        import torch.nn as nn
        class SimpleModel(nn.Module):
            def __init__(self, input_size, output_size, bias=False):
                super().__init__()
                self.classifier = torch.nn.Linear(input_size, output_size, bias=bias)

            def forward(self, x):
                return self.classifier(x)

        return SimpleModel(20, 30, bias)

    def test_model_no_bias(self):
        model_no_bias = self._create_model(bias=False)
        output_no_bias = model_no_bias(torch.randn(128, 20))
        print("\nOutput without bias:")
        print(output_no_bias)
        opt_foo = torch.compile(model_no_bias, backend='turbine_cpu')
        opt_foo(torch.randn(128, 20))

    def test_model_with_bias(self):
        model_with_bias = self._create_model(bias=True)
        output_with_bias = model_with_bias(torch.randn(128, 20))
        print("\nOutput with bias:")
        print(output_with_bias)
        opt_foo = torch.compile(model_with_bias, backend='turbine_cpu')
        opt_foo(torch.randn(128, 20))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
