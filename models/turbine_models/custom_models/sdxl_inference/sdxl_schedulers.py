# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# from @aviator19941's gist : https://gist.github.com/aviator19941/4e7967bd1787c83ee389a22637c6eea7

import os
import sys

from iree import runtime as ireert
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from diffusers import UNet2DConditionModel
from shark_turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)

import safetensors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token",
    type=str,
    help="The Hugging Face auth token, required",
    default=None,
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="stabilityai/stable-diffusion-xl-base-1.0",
)
parser.add_argument(
    "--scheduler_id",
    type=str,
    help="Scheduler ID",
    default="PNDM",
)
parser.add_argument(
    "--num_inference_steps", type=int, default=30, help="Number of inference steps"
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for inference"
)
parser.add_argument(
    "--height", type=int, default=1024, help="Height of Stable Diffusion"
)
parser.add_argument("--width", type=int, default=1024, help="Width of Stable Diffusion")
parser.add_argument(
    "--precision", type=str, default="fp32", help="Precision of Stable Diffusion"
)
parser.add_argument(
    "--max_length", type=int, default=77, help="Sequence Length of Stable Diffusion"
)
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument("--external_weight_path", type=str, default="")
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="saves ir/vmfb without global weights for size and readability, options [safetensors]",
)
parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, vulkan, rocm")
# TODO: Bring in detection for target triple
parser.add_argument(
    "--iree_target_triple",
    type=str,
    default="",
    help="Specify vulkan target triple or rocm/cuda target device.",
)
parser.add_argument("--vulkan_max_allocation", type=str, default="4294967296")


class SDXLScheduler(torch.nn.Module):
    def __init__(self, hf_model_name, num_inference_steps, scheduler, hf_auth_token=None, precision="fp32"):
        super().__init__()
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        self.guidance_scale = 7.5
        if precision == "fp16":
            try:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                    variant="fp16",
                )
            except:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                )
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                hf_model_name,
                subfolder="unet",
                auth_token=hf_auth_token,
                low_cpu_mem_usage=False,
            )

    def forward(
        self, sample, prompt_embeds, text_embeds, time_ids
    ):
        sample = sample * self.scheduler.init_noise_sigma
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": time_ids,
                }
                latent_model_input = torch.cat([sample] * 2)
                t = t.unsqueeze(0)
                # print('UNSQUEEZE T:', t)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, timestep=t
                )
                noise_pred = self.unet.forward(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                sample = self.scheduler.step(noise_pred, t, sample, return_dict=False)[0]
        return sample


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


    decomp_list = DEFAULT_DECOMPOSITIONS

    decomp_list.extend(
        [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
        ]
    )
    # encoder_hidden_states_sizes = (2, 77, 768)
    # if hf_model_name == "stabilityai/stable-diffusion-2-1-base":
    #     encoder_hidden_states_sizes = (2, 77, 1024)

    # tensor shapes for tracing
    # sample = torch.randn(1, 4, 128, 128)
    sample = (batch_size, 4, height // 8, width // 8)
    prompt_embeds = (2, 77, 2048)
    text_embeds = (2, 1280)
    time_ids = (2, 6)

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
            prompt_embeds=AbstractTensor(*prompt_embeds, dtype=torch.float32),
            text_embeds = AbstractTensor(*text_embeds, dtype=torch.float32), 
            time_ids = AbstractTensor(*time_ids, dtype=torch.float32),
        ):
            return jittable(scheduler.forward, decompose_ops=decomp_list)(sample, prompt_embeds, text_embeds, time_ids)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledScheduler(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))

    safe_name = utils.create_safe_name(hf_model_name, "-scheduler")
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(module_str)
    print("Saved to", safe_name + ".mlir")

    if compile_to != "vmfb":
        return module_str
    else:
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)


if __name__ == "__main__":
    args = parser.parse_args()
    hf_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    schedulers = utils.get_schedulers(args.hf_model_name)
    scheduler = schedulers[args.scheduler_id]
    scheduler_module = SDXLScheduler(args.hf_model_name, args.num_inference_steps, scheduler, hf_auth_token=None, precision=args.precision)

    print("export scheduler begin")
    mod_str = export_scheduler(
        scheduler_module,
        args.hf_model_name,
        args.batch_size,
        args.height,
        args.width,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.vulkan_max_allocation,
    )
    print("export scheduler complete")
    safe_name = utils.create_safe_name(args.hf_model_name, "-scheduler")
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
