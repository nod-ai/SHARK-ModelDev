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
from turbine_models.custom_models.pipeline_base import (
    PipelineComponent,
    TurbinePipelineBase,
)
from shark_turbine.transforms.general.add_metadata import AddMetadataPass

model_metadata_forward = {
    "model_name": "TestModel2xLinear",
    "input_shapes": [10],
    "input_dtypes": ["float32"],
    "output_shapes": [10],
    "output_dtypes": ["float32"],
    "test_kwarg_1": "test_kwarg_1_value",
    "test_kwarg_2": "test_kwarg_2_value",
}


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

    dummy_input = torch.empty(10)
    safe_keys = [
        model_metadata_forward["model_name"],
        "fp32",
        "bs1",
    ]
    safe_name = "_".join(safe_keys)
    vmfb_path = f"./{safe_name}.vmfb"

    fxb = FxProgramsBuilder(model)

    @fxb.export_program(args=(dummy_input,))
    def _forward(module, inputs):
        return module.forward(inputs)

    class CompiledTester(CompiledModule):
        forward = _forward

    inst = CompiledTester(context=Context(), import_to="IMPORT")
    mlir_module = CompiledModule.get_mlir_module(inst)
    mlir_module = AddMetadataPass(mlir_module, model_metadata_forward, "forward").run()
    vmfb_path = utils.compile_to_vmfb(
        str(mlir_module),
        device,
        target,
        None,
        safe_name + "_" + target,
        return_path=True,
    )
    return vmfb_path


class TestPipeline(TurbinePipelineBase):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def run(self, inputs: list):
        return self.test_model_1("forward", *inputs)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        model_map = {
            "test_model_1": {
                "model_name": "TestModel1",
                "external_weights": None,
                "module_name": "compiled_tester",
                "safe_name": "TestModel2xLinear",
                "keywords": ["Test", "Model", "2x", "Linear"],
                "export_fn": export_dummy_model,
                "export_args": None,
            }
        }
        self.pipe = TestPipeline(
            model_map=model_map,
            batch_size=1,
            device="cpu",
            iree_target_triple="x86_64-unknown-linux-gnu",
            pipeline_dir="./",
            precision="fp32",
        )
        self.pipe.prepare_all()
        self.pipe.load_map()
        self.test_input = [torch.ones(10)]

    def test_pipeline(self):
        output = self.pipe.run(self.test_input).to_host()
        print(output)

    def test_pipeline_benchmark(self):
        self.pipe.test_model_1.benchmark = True
        output = self.pipe.run(self.test_input).to_host()
        print(output)

    def test_pipeline_metadata(self):
        metadata = self.pipe.test_model_1.get_metadata("forward")
        expected = model_metadata_forward
        for i in expected.keys():
            expected[i] = str(expected[i])
        assert expected == metadata, "Metadata mismatch: expected {}, got {}".format(
            expected, metadata
        )


if __name__ == "__main__":
    unittest.main()
