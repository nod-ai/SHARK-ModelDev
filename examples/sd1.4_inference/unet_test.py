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
from diffusers import UNet2DConditionModel

pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

class UnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        self.guidance_scale = 7.5

    def forward(self, sample, timestep, encoder_hidden_states):
        samples = torch.cat([sample] * 2)
        unet_out = self.unet.forward(samples, timestep, encoder_hidden_states, return_dict=False)[0]
        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred


unet_model = UnetModel()


class CompiledUnet(aot.CompiledModule):
    params = aot.export_parameters(unet_model)

    def main(self, sample=aot.AbstractTensor(1, 4, 64, 64, dtype=torch.float32),
                timestep=aot.AbstractTensor(1, dtype=torch.float32),
                encoder_hidden_states=aot.AbstractTensor(2, 77, 768, dtype=torch.float32)):
        return aot.jittable(unet_model.forward)(
            sample, timestep, encoder_hidden_states
        )

exported = aot.export(CompiledUnet)
compiled_binary = exported.compile(save_to=None)
sample = np.random.rand(1, 4, 64, 64).astype(np.float32)
timestep = np.zeros((1)).astype(np.float32)
encoder_hidden_states = np.random.rand(2, 77, 768).astype(np.float32)
torch_sample = torch.from_numpy(sample)
torch_timestep = torch.from_numpy(timestep)
torch_encoder_hidden_states = torch.from_numpy(encoder_hidden_states)


def infer():
    import iree.runtime as rt

    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    output = vmm.main(sample, timestep, encoder_hidden_states)
    print(output.to_host(), output.to_host().shape, output.to_host().dtype)
    return output.to_host()


def infer_torch():
    torch_output = unet_model.forward(torch_sample, torch_timestep, torch_encoder_hidden_states)
    np_torch_output = torch_output.detach().cpu().numpy()
    print(np_torch_output, np_torch_output.shape, np_torch_output.dtype)
    return np_torch_output


def largest_error(array1, array2):
    absolute_diff = np.abs(array1 - array2)
    max_error = np.max(absolute_diff)
    return max_error


class ModelTests(unittest.TestCase):
    def testUnet(self):
        torch_output = infer_torch()
        turbine_output = infer()
        err = largest_error(torch_output, turbine_output)
        print('LARGEST ERROR:', err)
        assert(err < 9e-5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()