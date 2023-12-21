# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree import runtime as ireert
import iree.compiler as ireec
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils, unet
import torch
import torch._dynamo as dynamo
from diffusers import (
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    UNet2DConditionModel,
)
from torch.fx.experimental.proxy_tensor import make_fx


hf_model_name = "CompVis/stable-diffusion-v1-4"


class Scheduler(torch.nn.Module):
    def __init__(self, hf_model_name, num_inference_steps):
        super().__init__()
        self.scheduler = PNDMScheduler.from_pretrained(hf_model_name, subfolder="scheduler")
        self.scheduler.set_timesteps(num_inference_steps)
        self.unet = UNet2DConditionModel.from_pretrained(
            hf_model_name,
            subfolder="unet",
        )
        self.guidance_scale = 7.5

    def forward(self, latents, encoder_hidden_states):
        latents = latents * self.scheduler.init_noise_sigma
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            unet_out = self.unet.forward(
                latent_model_input, t, encoder_hidden_states, return_dict=False
            )[0]
            noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents

scheduler = Scheduler(hf_model_name, 10)
inputs = (torch.randn(1, 4, 64, 64), torch.randn(2, 77, 768))

fx_g = make_fx(
    scheduler,
    decomposition_table={},
    tracing_mode="symbolic",
    _allow_non_fake_inputs=True,
    _allow_fake_constant=False,
)(*inputs)

print(fx_g)