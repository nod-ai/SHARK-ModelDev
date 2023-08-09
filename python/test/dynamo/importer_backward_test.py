# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import struct
import unittest

from shark_turbine.dynamo.importer import FxImporter
import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd
from torch.fx import (
    GraphModule,
)
from torch.func import grad


class ImportTests(unittest.TestCase):
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

    def testImportCustomLossModule(self):
        def foo(x, y):
            loss = ((0.5 * x - y) ** 2).mean()
            loss.backward()
            return loss

        opt_foo = torch.compile(foo, backend=self.create_backend())
        opt_foo(torch.randn(10), torch.randn(10, requires_grad=True))

    # TODO: using func.grad for backward test

    # TODO: MNIST Classifier using LeNet for backward test


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
