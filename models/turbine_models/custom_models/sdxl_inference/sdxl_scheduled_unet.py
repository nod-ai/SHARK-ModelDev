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


class SDXLScheduledUnet(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        scheduler_id,
        height,
        width,
        batch_size,
        hf_auth_token=None,
        precision="fp32",
        num_inference_steps=1,
    ):
        super().__init__()
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.scheduler = utils.get_schedulers(hf_model_name)[scheduler_id]
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.is_scale_input_called = True

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

    def initialize(self, sample):
        height = sample.shape[-2] * 8
        width = sample.shape[-1] * 8
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
        add_time_ids = add_time_ids.repeat(sample.shape[0], 1).type(self.dtype)
        timesteps = self.scheduler.timesteps
        step_indexes = torch.tensor(len(timesteps))
        return sample * self.scheduler.init_noise_sigma, add_time_ids, step_indexes

    def forward(
        self, sample, prompt_embeds, text_embeds, time_ids, guidance_scale, step_index
    ):
        with torch.no_grad():
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
            t = self.scheduler.timesteps[step_index]
            sample = self.scheduler.scale_model_input(sample, t)
            latent_model_input = torch.cat([sample] * 2)
            noise_pred = self.unet.forward(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            sample = self.scheduler.step(noise_pred, t, sample, return_dict=False)[0]
        return sample


def export_scheduled_unet_model(
    scheduled_unet_model,
    scheduler_id,
    num_inference_steps,
    hf_model_name,
    batch_size,
    height,
    width,
    precision,
    max_length,
    hf_auth_token,
    compile_to,
    external_weights,
    external_weight_path,
    device,
    iree_target_triple,
    ireec_flags=None,
    decomp_attn=False,
    exit_on_vmfb=False,
    pipeline_dir=None,
):
    mapper = {}

    decomp_list = DEFAULT_DECOMPOSITIONS
    if decomp_attn == True:
        decomp_list.extend(
            [
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
                torch.ops.aten._scaled_dot_product_flash_attention.default,
            ]
        )

    dtype = torch.float16 if precision == "fp16" else torch.float32

    if precision == "fp16":
        scheduled_unet_model = scheduled_unet_model.half()

    utils.save_external_weights(
        mapper, scheduled_unet_model, external_weights, external_weight_path
    )

    sample = (
        batch_size,
        scheduled_unet_model.unet.config.in_channels,
        height // 8,
        width // 8,
    )
    prompt_embeds_shape = (2 * batch_size, max_length, 2048)
    text_embeds_shape = (2 * batch_size, 1280)
    time_ids_shape = (2 * batch_size, 6)

    class CompiledScheduledUnet(CompiledModule):
        if external_weights:
            params = export_parameters(
                scheduled_unet_model,
                external=True,
                external_scope="",
                name_mapper=mapper.get,
            )
        else:
            params = export_parameters(scheduled_unet_model)

        def run_initialize(
            self,
            sample=AbstractTensor(*sample, dtype=dtype),
        ):
            return jittable(scheduled_unet_model.initialize)(sample)

        def run_forward(
            self,
            sample=AbstractTensor(*sample, dtype=dtype),
            prompt_embeds=AbstractTensor(*prompt_embeds_shape, dtype=dtype),
            text_embeds=AbstractTensor(*text_embeds_shape, dtype=dtype),
            time_ids=AbstractTensor(*time_ids_shape, dtype=dtype),
            guidance_scale=AbstractTensor(1, dtype=dtype),
            step_index=AbstractTensor(1, dtype=torch.int64),
        ):
            return jittable(scheduled_unet_model.forward, decompose_ops=decomp_list)(
                sample, prompt_embeds, text_embeds, time_ids, guidance_scale, step_index
            )

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledScheduledUnet(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    if pipeline_dir:
        safe_name = os.path.join(
            pipeline_dir, f"{scheduler_id}_unet_{str(num_inference_steps)}"
        )
    else:
        safe_name = utils.create_safe_name(
            hf_model_name,
            f"_{max_length}_{height}x{width}_{precision}_scheduled_unet_{device}",
        )
    if compile_to != "vmfb":
        return module_str
    elif os.path.isfile(safe_name + ".vmfb") and exit_on_vmfb:
        exit()
    elif compile_to == "vmfb":
        vmfb = utils.compile_to_vmfb(
            module_str,
            device,
            iree_target_triple,
            ireec_flags,
            safe_name,
            return_path=True,
        )
        if exit_on_vmfb:
            exit()
        return vmfb


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    scheduled_unet_model = SDXLScheduledUnet(
        args.hf_model_name,
        args.scheduler_id,
        args.height,
        args.width,
        args.batch_size,
        args.hf_auth_token,
        args.precision,
        args.num_inference_steps,
    )
    mod_str = export_scheduled_unet_model(
        scheduled_unet_model,
        args.scheduler_id,
        args.num_inference_steps,
        args.hf_model_name,
        args.batch_size,
        args.height,
        args.width,
        args.precision,
        args.max_length,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.ireec_flags,
        args.decomp_attn,
        args.exit_on_vmfb,
        args.pipeline_dir,
    )
    safe_name = utils.create_safe_name(
        args.hf_model_name + "_" + args.scheduler_id,
        f"_{args.max_length}_{args.height}x{args.width}_{args.precision}_unet_{str(args.num_inference_steps)}",
    )
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
