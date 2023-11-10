# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import shark_turbine.aot as aot
import torch
import torch._dynamo as dynamo
from torch._export import dynamic_dim
from torch._export.constraints import constrain_as_size, constrain_as_value
from diffusers import AutoencoderKL

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"


class VaeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae"
        )

    def forward(self, inp):
        x = self.vae.decode(inp, return_dict=False)[0]
        return x


vae_model = VaeModel()
example_x = torch.empty(1, 4, 64, 64, dtype=torch.float32)
exported = aot.export(vae_model, example_x)
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
    inp = np.random.rand(1, 4, 64, 64).astype(np.float32)
    output = vmm.main(inp)
    print(output.to_host())


class ModelTests(unittest.TestCase):
    def testVAE(self):
        infer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()