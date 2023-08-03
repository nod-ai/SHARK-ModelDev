# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

from shark_turbine.dynamo.importer import FxImporter
import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd
from torch.fx import (
    GraphModule,
)


class TensorScalarOpConversionImportModule(unittest.TestCase):
    def setUp(self):
        self.t = torch.randn(2, 2)

    def add(self, x):
        return x + 8.2

    def sub(self, x):
        return x - 1.6

    def mul(self, x):
        return x * 3.2

    def div(self, x):
        return x / 2.1

    def floor_div(self, x):
        return x // 4.2

    def create_backend(self):
        imp = FxImporter()

        def import_compiler(gm: GraphModule, example_inputs):
            gm.print_readable()
            try:
                imp.import_graph_module(gm)
            finally:
                print(imp.module)
            imp.module.operation.verify()
            return gm

        backend = import_compiler
        backend = aot_autograd(fw_compiler=backend)
        return backend

    def test_add(self):
        opt_torch_scalar_convert = torch.compile(
            self.add, backend=self.create_backend()
        )
        result = opt_torch_scalar_convert(self.t)
        expected_result = self.add(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def test_sub(self):
        opt_torch_scalar_convert = torch.compile(
            self.sub, backend=self.create_backend()
        )
        result = opt_torch_scalar_convert(self.t)
        expected_result = self.sub(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def test_mul(self):
        opt_torch_scalar_convert = torch.compile(
            self.mul, backend=self.create_backend()
        )
        result = opt_torch_scalar_convert(self.t)
        expected_result = self.mul(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def test_div(self):
        opt_torch_scalar_convert = torch.compile(
            self.div, backend=self.create_backend()
        )
        result = opt_torch_scalar_convert(self.t)
        expected_result = self.div(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def test_floor_div(self):
        """
        This op isn't successfully created by IREE due to partial implementation of floor_div op in torch-mlir
        However, the importer works successfully.
        """
        opt_torch_scalar_convert = torch.compile(
            self.floor_div, backend=self.create_backend()
        )
        result = opt_torch_scalar_convert(self.t)
        expected_result = self.floor_div(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
