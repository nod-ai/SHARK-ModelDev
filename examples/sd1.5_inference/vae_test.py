# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import numpy as np
import shark_turbine.aot as aot
import torch
import torch._dynamo as dynamo
from torch._export import dynamic_dim
from torch._export.constraints import constrain_as_size, constrain_as_value
from diffusers import AutoencoderKL

pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"


class VaeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae"
        )

    def forward(self, inp):
        with torch.no_grad():
            x = self.vae.decode(inp, return_dict=False)[0]
            return x


vae_model = VaeModel()


class CompiledUnet(aot.CompiledModule):
    params = aot.export_parameters(vae_model)

    def main(self, inp=aot.AbstractTensor(1, 4, 64, 64, dtype=torch.float32)):
        return aot.jittable(vae_model.forward)(
            inp
        )


exported = aot.export(CompiledUnet)
compiled_binary = exported.compile(save_to=None)
inp = np.random.rand(1, 4, 64, 64).astype(np.float32)
torch_inp = torch.from_numpy(inp)


def infer():
    import iree.runtime as rt

    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    output = vmm.main(inp)
    print(output.to_host(), output.to_host().shape, output.to_host().dtype)
    return output.to_host()


def infer_torch():
    torch_output = vae_model.forward(torch_inp)
    torch_output = torch_output.detach().cpu().numpy()
    print(torch_output, torch_output.shape, torch_output.dtype)
    return torch_output


def largest_error(array1, array2):
    absolute_diff = np.abs(array1 - array2)
    max_error = np.max(absolute_diff)
    return max_error


class ModelTests(unittest.TestCase):
    def testVAE(self):
        turbine_output = infer()
        torch_output = infer_torch()
        err = largest_error(torch_output, turbine_output)
        print('LARGEST ERROR:', err)
        assert(err < 8e-5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()