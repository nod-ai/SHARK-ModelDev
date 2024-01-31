# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from shark_turbine.aot import *
from iree import runtime as ireert
import iree.compiler as ireec
from iree.compiler.ir import Context
import numpy as np

from turbine_models.custom_models.sd_inference import utils
from diffusers import (
    PNDMScheduler,
    UNet2DConditionModel,
)


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

    def forward(self, latents, encoder_hidden_states) -> torch.FloatTensor:
        latents = latents * self.scheduler.init_noise_sigma
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            t = t.unsqueeze(0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            unet_out = self.unet.forward(
                latent_model_input, t, encoder_hidden_states, return_dict=False
            )[0]
            noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents


def export_scheduler(
    scheduler,
    hf_model_name,
    batch_size,
    height,
    width,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target_triple=None,
    max_alloc=None,
):
    mapper = {}
    utils.save_external_weights(
        mapper, scheduler, external_weights, external_weight_path
    )

    encoder_hidden_states_sizes = (2, 77, 768)
    if hf_model_name == "stabilityai/stable-diffusion-2-1-base":
        encoder_hidden_states_sizes = (2, 77, 1024)

    sample = (batch_size, 4, height // 8, width // 8)

    class CompiledScheduler(CompiledModule):
        if external_weights:
            params = export_parameters(
                scheduler, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(scheduler)

        def main(
            self,
            sample=AbstractTensor(*sample, dtype=torch.float32),
            encoder_hidden_states=AbstractTensor(
                *encoder_hidden_states_sizes, dtype=torch.float32
            ),
        ):
            return jittable(scheduler.forward)(sample, encoder_hidden_states)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledScheduler(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = utils.create_safe_name(hf_model_name, "-scheduler)
    if compile_to != "vmfb":
        return module_str
    else:
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)


if __name__ == '__main__':
    hf_model_name = "CompVis/stable-diffusion-v1-4"
    scheduler = Scheduler(hf_model_name, 2)
    inputs = (torch.randn(1, 4, 64, 64), torch.randn(2, 77, 768),)
    batch_size = 1
    height = 512
    width = 512
    hf_auth_token = None
    compile_to = "vmfb"
    external_weights = None
    external_weight_path = "stable_diffusion_v1_4_clip.safetensors"
    device = "cpu"
    iree_target_triple = None
    vulkan_max_allocation = None

    mod_str = export_scheduler(
        scheduler,
        hf_model_name,
        batch_size,
        height,
        width,
        hf_auth_token,
        compile_to,
        external_weights,
        external_weight_path,
        device,
        iree_target_triple,
        vulkan_max_allocation,
    )
    safe_name = utils.create_safe_name(hf_model_name, "-vae")
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")