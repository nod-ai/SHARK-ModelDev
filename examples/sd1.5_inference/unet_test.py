# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import shark_turbine.aot as aot
import torch
from diffusers import UNet2DConditionModel


pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

class UnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet.forward(sample, timestep, encoder_hidden_states, return_dict=False)[0]


unet_model = UnetModel()


class CompiledUnet(aot.CompiledModule):
    params = aot.export_parameters(unet_model)

    def main(self, sample=aot.AbstractTensor(1, 4, 64, 64, dtype=torch.float32),
                timestep=aot.AbstractTensor(1, dtype=torch.float32),
                encoder_hidden_states=aot.AbstractTensor(1, 77, 768, dtype=torch.float32)):
        return aot.jittable(unet_model.forward)(
            sample, timestep, encoder_hidden_states
        )

exported = aot.export(CompiledUnet)
exported._run_import()
from contextlib import redirect_stdout
with open('unet_test2.mlir', 'w') as f:
    with redirect_stdout(f):
        exported.print_readable()
compiled_binary = exported.compile(save_to=None)

def infer():
    import numpy as np
    import iree.runtime as rt

    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    sample = np.random.rand(1, 4, 64, 64).astype(np.float32)
    timestep = np.ones((1)).astype(np.float32)
    encoder_hidden_states = np.random.rand(1, 77, 768).astype(np.float32)
    output = vmm.main(sample, timestep, encoder_hidden_states)
    print(output.to_host(), output.to_host().shape)


class ModelTests(unittest.TestCase):
    def testUnet(self):
        infer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()