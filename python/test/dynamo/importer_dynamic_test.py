# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import torch._dynamo as dynamo
from torch._export import dynamic_dim
# from torch._export.constraints import constrain_as_size, constrain_as_value
from shark_turbine.dynamo.importer import FxImporter
from shark_turbine.dynamo.passes import turbine_cpu_pass_pipeline
import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch.func import functionalize
from torch.fx import (
    GraphModule,
)

def import_compiler(gm: GraphModule, example_inputs, decompose_ops=None):
    imp = FxImporter()
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

class DynamicBMM(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight0 = torch.nn.Parameter(torch.rand(n, k))

    def forward(self, inp, *, bias):
        mm = torch.matmul(inp, self.weight0)
        biased = mm + bias
        return {"result": biased}

class DynamicBuiltinOps(torch.nn.Module):
    def forward(self, inp):
        x = inp.size()[1] - inp.size()[2]
        x = x * inp.size()[1] - 34.2
        g = x / 32
        return {"result": g}

class ProgramTests(unittest.TestCase):
    def testStaticExport(self):
        model = DynamicBMM(12, 19)
        inp_example = torch.rand(1, 2, 12)
        bias_example = torch.rand(19)
        f = dynamo.export(
            model.forward,
            aten_graph=True,
            same_signature=False,
            assume_static_by_default=True,
            constraints=[dynamic_dim(inp_example, 1) >= 2,],
        )
        g, guards = f(inp=inp_example, bias=bias_example)
        g = import_compiler(g, [inp_example, bias_example])

    def testStaticExportSameSignatureTrue(self):
        model = DynamicBMM(12, 19)
        inp_example = torch.rand(1, 2, 12)
        bias_example = torch.rand(19)
        f = dynamo.export(
            model.forward,
            aten_graph=True,
            same_signature=True,
            assume_static_by_default=True,
            constraints=[dynamic_dim(inp_example, 1) >= 2,],
        )
        g, guards = f(inp=inp_example, bias=bias_example)
        g = import_compiler(g, [inp_example, bias_example])

    def testStaticExportBuiltinOps(self):
        model = DynamicBuiltinOps()
        inp_example = torch.rand(1, 2, 12)
        f = dynamo.export(
            model.forward,
            aten_graph=True,
            same_signature=True,
            assume_static_by_default=True,
            constraints=[dynamic_dim(inp_example, 1) >= 2,],
        )
        g, guards = f(inp=inp_example)
        g = import_compiler(g, [inp_example])





if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()