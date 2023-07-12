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


class ImportTests(unittest.TestCase):
    def testInitialize(self):
        imp = FxImporter()
        print(imp.module)

    def testImportStateless(self):
        imp = FxImporter()

        def import_compiler(gm: GraphModule, example_inputs):
            gm.print_readable()
            try:
                imp.import_stateless_graph(gm.graph)
            finally:
                print(imp.module)
            imp.module.operation.verify()
            return gm
        backend = import_compiler
        backend = aot_autograd(fw_compiler=backend)

        a = torch.randn(3, 4)
        @dynamo.optimize(backend)
        def basic(x):
            return torch.tanh(x) * a
        basic(torch.randn(3, 4))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
