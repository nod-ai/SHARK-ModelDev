# Copyright 2024 Advanced Micro Devices, inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import unittest
import torch
import os
import numpy as np
from iree.compiler.ir import Context
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
from shark_turbine.transforms import FuncOpMatcher, Pass


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


torch.no_grad()


def export_dummy_model():
    model = TestModule()
    target = "x86_64-unknown-linux-gnu"
    device = "llvm-cpu"
    model_metadata = {
        "model_name": "TestModel2xLinear",
        "input_shapes": [(10,)],
        "input_dtypes": ["float32"],
        "output_shapes": [(10,)],
        "output_dtypes": ["float32"],
        "test_kwarg_1": "test_kwarg_1_value",
        "test_kwarg_2": "test_kwarg_2_value",
    }
    dummy_input = torch.empty(10)
    safe_name = model_metadata["model_name"].replace("/", "_")
    vmfb_path = f"./{safe_name}.vmfb"

    fxb = FxProgramsBuilder(model)

    @fxb.export_program(args=(dummy_input,))
    def _forward(module, inputs):
        return module.forward(inputs)

    class CompiledTester(CompiledModule):
        forward = _forward

    inst = CompiledTester(context=Context(), import_to="IMPORT")
    mlir_module = CompiledModule.get_mlir_module(inst)
    funcop_pass = Pass(mlir_module.operation)

    breakpoint()


# class PipelineTest(unittest.TestCase):
#     def setUp(self):
#         model_map = {
#             'test_model_1':
#         }

if __name__ == "__main__":
    export_dummy_model()
