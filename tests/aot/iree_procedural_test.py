# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch

from iree.compiler.ir import (
    Context,
)

from shark_turbine.aot import *


class CompiledModuleAPI(unittest.TestCase):
    def testTensorDim(self):
        class BasicModule(CompiledModule):
            def foobar(self, a=AbstractTensor(None, 3)):
                return IREE.tensor_dim(a, 0)

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("%c0 = arith.constant 0", module_str)
        self.assertIn("%dim = tensor.dim %arg0, %c0", module_str)
        self.assertIn("return %dim", module_str)

    def testTensorEmpty(self):
        class BasicModule(CompiledModule):
            def foobar(self, x=AbstractIndex):
                empty = IREE.tensor_empty(x, 16)
                dim0 = IREE.tensor_dim(empty, 0)
                return empty, dim0

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("%0 = flow.tensor.empty : tensor<?x16xf32>{%arg0}", module_str)
        # NOTE: We are testing below that the dynamic dimension is associated
        # and used from the input vs being recalculated.
        self.assertIn("return %0, %arg0 : tensor<?x16xf32>, index", module_str)

    def testTensorSplat(self):
        class BasicModule(CompiledModule):
            def foobar(self, x=AbstractIndex, y=AbstractF32):
                empty = IREE.tensor_splat(x, 34, value=y, dtype=torch.float32)
                dim0 = IREE.tensor_dim(empty, 0)
                return empty, dim0

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "%0 = flow.tensor.splat %arg1 : tensor<?x34xf32>{%arg0}", module_str
        )
        # NOTE: We are testing below that the dynamic dimension is associated
        # and used from the input vs being recalculated.
        self.assertIn("return %0, %arg0 : tensor<?x34xf32>, index", module_str)

    def testTensorTrace(self):
        class BasicModule(CompiledModule):
            def foobar(self, x=AbstractTensor(None), y=AbstractTensor(3)):
                IREE.tensor_trace("DEBUG", x, y)

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn('flow.tensor.trace {key = "DEBUG"} %arg0, %arg1', module_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
