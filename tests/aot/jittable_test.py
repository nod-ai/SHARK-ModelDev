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


class JittableTests(unittest.TestCase):
    def testImportPhases(self):
        class ExportedProcModule(CompiledModule):
            def foobar(self):
                return self.compute(), self.compute()

            @CompiledModule.jittable
            def compute():
                t1 = torch.ones(2, 2)
                t2 = t1 + t1
                return t2 * t2

        inst = ExportedProcModule(context=Context(), import_to="import")
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        # Functions should still be on torch types.
        self.assertIn(
            "func private @compute() -> !torch.vtensor<[2,2],f32>", module_str
        )
        CompiledModule.run_import(inst)
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertNotIn("!torch.vtensor", module_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
