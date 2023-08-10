# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
from typing import List

from shark_turbine.dynamo.importer import FxImporter
import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch.func import functionalize
from torch.fx import (
    GraphModule,
)


class ImportTests(unittest.TestCase):
    def create_backend(self, decompose_ops: List[torch._ops.OpOverloadPacket] = None):
        imp = FxImporter()

        def import_compiler(gm: GraphModule, example_inputs):
            if decompose_ops is not None:
                gm = make_fx(
                    functionalize(gm),
                    decomposition_table=get_decompositions(decompose_ops),
                )(*example_inputs)

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

    def testImportStateless(self):
        a = torch.randn(3, 4)
        backend = self.create_backend()

        @dynamo.optimize(backend)
        def basic(x):
            return torch.tanh(x) * a

        basic(torch.randn(3, 4))

    def testImportDtype(self):
        def foo(x):
            o = x.to(torch.complex32)
            o = o.to(torch.float32)
            o = o.to(torch.float64)
            o = o.to(torch.float16)
            o = o.to(torch.int64)
            o = o.to(torch.int32)
            o = o.to(torch.int16)
            o = o.to(torch.int8)
            o = o.to(torch.uint8)
            o = o.to(torch.complex64)
            o = o.to(torch.bool)
            # o = o.to(torch.qint8) # we do not currently support quantized dtypes
            # o = o.to(torch.quint8)
            o = o.to(torch.bfloat16)
            return o

        opt_foo = torch.compile(foo, backend=self.create_backend())
        opt_foo(torch.ones(10))

    def testImportDevice(self):
        def foo(x):
            return torch.arange(x, device="cpu")

        opt_foo = torch.compile(foo, backend=self.create_backend())
        opt_foo(10)

    def testImportLayout(self):
        def foo(x):
            # sparse layouts are not currently supported as they can not be created on the 'meta' device
            return torch.ones_like(x, layout=torch.strided)

        opt_foo = torch.compile(foo, backend=self.create_backend())
        opt_foo(torch.randn(10))

    def testImportMemoryFormat(self):
        def foo():
            x = torch.ones_like(torch.randn(10), memory_format=torch.contiguous_format)
            x = torch.ones_like(torch.randn(10), memory_format=torch.preserve_format)
            x = torch.ones_like(
                torch.randn(1, 1, 1, 1), memory_format=torch.channels_last
            )
            x = torch.ones_like(
                torch.randn(1, 1, 1, 1, 1), memory_format=torch.channels_last_3d
            )

        opt_foo = torch.compile(foo, backend=self.create_backend())
        opt_foo()

    def testImportListArgs(self):
        def foo():
            return torch.randn((4, 5, 6))

        opt_foo = torch.compile(foo, backend=self.create_backend())
        opt_foo()

    def testImportListNodeArgs(self):
        def foo(x, y):
            return torch.cat((x, y), 0)

        opt_foo = torch.compile(foo, backend=self.create_backend())
        opt_foo(torch.randn(10), torch.randn(10))

    def testImportDecomposeChunk(self):
        def foo_chunk(x):
            return torch.chunk(x, 2, dim=-1)

        opt = torch.compile(
            foo_chunk,
            backend=self.create_backend(
                decompose_ops=[
                    torch.ops.aten.split.Tensor,
                    torch.ops.aten.split_with_sizes,
                ]
            ),
        )
        t = torch.randn([4, 4, 4, 4])
        opt(t)

    def testImportDecomposeBatchNorm2D(self):
        def foo_chunk(x):
            return torch.nn.BatchNorm2d(4)(x)

        opt = torch.compile(
            foo_chunk,
            backend=self.create_backend(
                decompose_ops=[
                    torch.ops.aten._native_batch_norm_legit_functional,
                    torch.ops.aten.squeeze.dims,
                ]
            ),
        )
        t = torch.randn([4, 4, 4, 4])
        opt(t)

    @unittest.expectedFailure
    def testImportToList(self):
        """
        Marked as XFail due to No stacktrace found for _tensor_constant0 = self._tensor_constant0
        that leads to copy of None in aten.lift_fresh_copy op and fails to get operands
        """

        def foo():
            tensor_data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            list_data = tensor_data.tolist()
            return list_data

        opt_foo = torch.compile(foo, backend=self.create_backend())
        result = opt_foo()
        expected_result = foo()
        self.assertEqual(result, expected_result, "broken")

    @unittest.expectedFailure
    def testImportStandardList(self):
        """
        Marked as XFail due to No stacktrace found for _tensor_constant0 = self._tensor_constant0
        that leads to copy of None in aten.lift_fresh_copy op and fails to get operands
        """

        def foo():
            tensor_data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            list_data = list(tensor_data)
            return list_data

        opt_foo = torch.compile(foo, backend=self.create_backend())
        opt_foo()

    def testImportVisionModule(self):
        from torch import nn
        import torch.nn.functional as F

        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
                super(ConvBlock, self).__init__()
                self.stride = stride
                self.channel_pad = out_channels - in_channels
                padding = (kernel_size - 1) // 2
                self.convs = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=in_channels,
                        bias=True,
                    ),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    ),
                )
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                h = x
                if self.channel_pad > 0:
                    x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
                return self.act(self.convs(h) + x)

        mod = ConvBlock(3, 5)
        opt_mod = torch.compile(mod, backend=self.create_backend())
        opt_mod(torch.randn(1, 3, 256, 256))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
