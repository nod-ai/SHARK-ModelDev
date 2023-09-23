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
            def foobar(self, a: AbstractTensor(None, 3)):
                return IREE.tensor_dim(a, 0)

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("%c0 = arith.constant 0", module_str)
        self.assertIn("%dim = tensor.dim %arg0, %c0", module_str)
        self.assertIn("return %dim", module_str)

    def testTensorEmpty(self):
        class BasicModule(CompiledModule):
            def foobar(self, x: AbstractIndex):
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
            def foobar(self, x: AbstractIndex, y: AbstractF32):
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
            def foobar(self, x: AbstractTensor(None), y: AbstractTensor(3)):
                IREE.tensor_trace("DEBUG", x, y)

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn('flow.tensor.trace {key = "DEBUG"} %arg0, %arg1', module_str)

    def testStoreDynamic(self):
        class BasicModule(CompiledModule):
            x = export_global(AbstractTensor(None, 34), mutable=True)

            def foobar(self, x: AbstractIndex, y: AbstractF32):
                splat = IREE.tensor_splat(x, 34, value=y, dtype=torch.float32)
                self.x = splat

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "util.global private mutable @_x.global {noinline} : tensor<?x34xf32>",
            module_str,
        )
        self.assertIn("%0 = flow.tensor.splat", module_str)
        self.assertIn("util.global.store %0, @_x.global : tensor<?x34xf32>", module_str)

    def testTensorSliceStatic(self):
        class BasicModule(CompiledModule):
            def foobar(self, x: AbstractTensor(3, 4)):
                return IREE.tensor_slice(x, 0, (1, 3))

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "flow.tensor.slice %arg0[%c0, %c1_0 for %c1, %c3] : tensor<3x4xf32> -> tensor<1x3xf32>",
            module_str,
        )

    def testTensorSliceDynamicIndex(self):
        class SliceDynamicIndex(CompiledModule):
            def foobar(self, x: AbstractIndex):
                empty = IREE.tensor_empty(x, 16)
                return IREE.tensor_slice(empty, x, 4)

        inst = SliceDynamicIndex(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "flow.tensor.slice %0[%arg0, %c4 for %c1, %c1] : tensor<?x16xf32>{%arg0} -> tensor<1x1xf32>",
            module_str,
        )

    def testTensorSliceDynamicLength(self):
        class SliceDynamicIndex(CompiledModule):
            def foobar(self, x: AbstractIndex, y: AbstractIndex):
                empty = IREE.tensor_empty(x, 16)
                return IREE.tensor_slice(empty, (x, y), 4)

        inst = SliceDynamicIndex(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "flow.tensor.slice %0[%arg0, %c4 for %arg1, %c1] : tensor<?x16xf32>{%arg0} -> tensor<?x1xf32>{%arg1}",
            module_str,
        )

    def testTensorUpdateStatic(self):
        class UpdateStatic(CompiledModule):
            def foobar(
                self,
                target: AbstractTensor(4, 4),
                update: AbstractTensor(2, 2),
                i: AbstractIndex,
                j: AbstractIndex,
            ):
                return IREE.tensor_update(target, update, i, j)

        inst = UpdateStatic(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "flow.tensor.update %arg1, %arg0[%arg2, %arg3] : tensor<2x2xf32> -> %arg0 as tensor<4x4xf32>",
            module_str,
        )

    def testTensorUpdateDynamic(self):
        class UpdateDynamic(CompiledModule):
            def foobar(
                self,
                x: AbstractIndex,
                y: AbstractIndex,
                i: AbstractIndex,
                j: AbstractIndex,
                value: AbstractF32,
            ):
                target = IREE.tensor_empty(x, y)
                update = IREE.tensor_splat(i, j, value=value, dtype=torch.float32)
                return IREE.tensor_update(target, update, 2, 2)

        inst = UpdateDynamic(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "flow.tensor.update %1, %0[%c2, %c2] : tensor<?x?xf32>{%arg2, %arg3} -> %0 as tensor<?x?xf32>{%arg0, %arg1}",
            module_str,
        )

    def testTensorReshape(self):
        class ReshapeModule(CompiledModule):
            def foobar(self, x: AbstractIndex, y: AbstractIndex):
                empty = IREE.tensor_empty(x, 16)
                reshaped = IREE.tensor_reshape(empty, 1, y, y)
                return reshaped

        inst = ReshapeModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "flow.tensor.reshape %0 : tensor<?x16xf32>{%arg0} -> tensor<1x?x?xf32>{%arg1, %arg1}",
            module_str,
        )

    def testScalarAddInt(self):
        class ArithModule(CompiledModule):
            def foobar(self, a: AbstractI32, b: AbstractI32):
                return a + b

        inst = ArithModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        self.assertIn("arith.addi %arg0, %arg1 : i32", module_str)

    def testScalarAddFloat(self):
        class ArithModule(CompiledModule):
            def foobar(self, a: AbstractF32, b: AbstractF32):
                return a + b

        inst = ArithModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        self.assertIn("arith.addf %arg0, %arg1 : f32", module_str)

    def testScalarAddLiteral(self):
        class ArithModule(CompiledModule):
            def foobar(self, a: AbstractI32):
                return a + 1

        inst = ArithModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        self.assertIn("%c1_i32 = arith.constant 1 : i32", module_str)
        self.assertIn("arith.addi %arg0, %c1_i32 : i32", module_str)

    def testScalarAddLiteralMixedType(self):
        class ArithModule(CompiledModule):
            def foobar(self, a: AbstractI32):
                return a + 3.23

        inst = ArithModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        self.assertIn("%0 = arith.sitofp %arg0 : i32 to f32", module_str)
        self.assertIn("%cst = arith.constant 3.230000e+00 : f32", module_str)
        self.assertIn("arith.addf %0, %cst : f32", module_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
