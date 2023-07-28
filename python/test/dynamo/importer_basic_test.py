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
                imp.import_graph_module(gm)
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

    def testImportVisionModule(self):
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

        from torch import nn
        import torch.nn.functional as F
        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
                super(ConvBlock, self).__init__()
                self.stride = stride
                self.channel_pad = out_channels - in_channels
                padding = (kernel_size - 1) // 2
                self.convs = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=True), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True))
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                h = x
                if self.channel_pad > 0:
                    x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), 'constant', 0)
                return self.act(self.convs(h) + x)

        mod = ConvBlock(3,5)
        opt_mod = torch.compile(mod, backend=backend)
        opt_mod(torch.randn(1,3,256,256))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
