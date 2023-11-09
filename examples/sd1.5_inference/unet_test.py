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
from PIL import Image


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
exported.print_readable()

print('DONE')