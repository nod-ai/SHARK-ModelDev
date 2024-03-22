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
from shark_turbine.aot.builtins import *


class TorchExportTests(unittest.TestCase):
    def testImportPhases(self):
        class MyModule(torch.nn.Module):
            def forward(self):
                ...

        fxb = FxProgramsBuilder(MyModule())

        @fxb.export_program(
            args=([torch.empty([3, 2]), torch.empty([1, 2])],),
            kwargs={"foobar": torch.empty([3, 1])},
        )
        def compute(module, inputs, *, foobar):
            t1 = inputs[0]
            t2 = inputs[1]
            t3 = t1 + t2 + foobar
            return [t3 * t3, foobar]

        class ExportedProcModule(CompiledModule):
            _compute = compute

            def foobar(
                self,
                t1=AbstractTensor(3, 2),
                t2=AbstractTensor(1, 2),
                t3=AbstractTensor(3, 1),
            ):
                return self._compute(t1, t2, foobar=t3)

        inst = ExportedProcModule(context=Context(), import_to="import")
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
