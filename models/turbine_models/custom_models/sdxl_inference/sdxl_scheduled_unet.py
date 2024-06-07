# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# from @aviator19941's gist : https://gist.github.com/aviator19941/4e7967bd1787c83ee389a22637c6eea7

import copy
import os
import sys
import numpy as np

# os.environ["TORCH_LOGS"] = "+dynamo"

import torch
import torch._dynamo as dynamo

from iree import runtime as ireert
from iree.compiler.ir import Context

from shark_turbine.aot import *
import shark_turbine.ops as ops

from turbine_models.custom_models.sd_inference import utils
from turbine_models.custom_models.sd_inference.schedulers import get_scheduler
from diffusers import UNet2DConditionModel


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
        return_index=False,
    ):
        super().__init__()
        self.do_classifier_free_guidance = True
        # if any(key in hf_model_name for key in ["turbo", "lightning"]):
        #     self.do_classifier_free_guidance = False
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.scheduler = utils.get_schedulers(hf_model_name)[scheduler_id]
        # if scheduler_id == "PNDM":
        #     num_inference_steps = num_inference_steps - 1
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.is_scale_input_called = True
        self.return_index = return_index
        self.height = height
        self.width = width
        self.batch_size = batch_size

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
        height = self.height
        width = self.width
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype)
        if self.do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            add_time_ids = add_time_ids.repeat(self.batch_size, 1).type(self.dtype)
        timesteps = self.scheduler.timesteps
        step_indexes = torch.tensor(len(timesteps))
        sample = sample * self.scheduler.init_noise_sigma
        return sample.type(self.dtype), add_time_ids, step_indexes

    def forward(
        self, sample, prompt_embeds, text_embeds, time_ids, guidance_scale, step_index
    ):
        added_cond_kwargs = {
            "time_ids": time_ids,
            "text_embeds": text_embeds,
        }
        t = self.scheduler.timesteps[step_index]
        if self.do_classifier_free_guidance:
            latent_model_input = torch.cat([sample] * 2)
        else:
            latent_model_input = sample
        # ops.iree.trace_tensor(f"latent_model_input_{step_index}", latent_model_input)

        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, t
        ).type(self.dtype)
        print(
            latent_model_input.shape,
            t.shape,
            sample.shape,
            prompt_embeds.shape,
            added_cond_kwargs,
            guidance_scale,
            step_index,
        )
        # ops.iree.trace_tensor(f"latent_model_input_scaled_{step_index}", latent_model_input)
        noise_pred = self.unet.forward(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        # ops.iree.trace_tensor(f"noise_pred_{step_index}", noise_pred)

        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        sample = self.scheduler.step(noise_pred, t, sample, return_dict=False)[0]
        return sample.type(self.dtype)


@torch.no_grad()
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
    attn_spec=None,
    input_mlir=None,
    weights_only=False,
):
    # if "turbo" in hf_model_name:
    #     do_classifier_free_guidance = False
    # else:
    #     do_classifier_free_guidance = True
    do_classifier_free_guidance = True
    if pipeline_dir:
        safe_name = os.path.join(
            pipeline_dir, f"{scheduler_id}_unet_{str(num_inference_steps)}"
        )
    else:
        safe_name = utils.create_safe_name(
            hf_model_name,
            f"_bs{batch_size}_{max_length}_{height}x{width}_{precision}_scheduled_unet_{device}",
        )

    if input_mlir:
        vmfb_path = utils.compile_to_vmfb(
            input_mlir,
            device,
            iree_target_triple,
            ireec_flags,
            safe_name,
            mlir_source="file",
            return_path=not exit_on_vmfb,
            attn_spec=attn_spec,
        )
        return vmfb_path

    dtype = torch.float16 if precision == "fp16" else torch.float32

    if precision == "fp16":
        scheduled_unet_model = scheduled_unet_model.half()

    mapper = {}
    utils.save_external_weights(
        mapper, scheduled_unet_model, external_weights, external_weight_path
    )
    if weights_only:
        return external_weight_path

    if do_classifier_free_guidance:
        init_batch_dim = 2
    else:
        init_batch_dim = 1

    sample_shape = [
        batch_size,
        scheduled_unet_model.unet.config.in_channels,
        height // 8,
        width // 8,
    ]
    time_ids_shape = [init_batch_dim * batch_size, 6]
    prompt_embeds_shape = [init_batch_dim * batch_size, max_length, 2048]
    text_embeds_shape = [init_batch_dim * batch_size, 1280]

    fxb = FxProgramsBuilder(scheduled_unet_model)

    example_init_args = [torch.empty(sample_shape, dtype=dtype)]
    example_forward_args = [
        torch.empty(sample_shape, dtype=dtype),
        torch.empty(prompt_embeds_shape, dtype=dtype),
        torch.empty(text_embeds_shape, dtype=dtype),
        torch.empty(time_ids_shape, dtype=dtype),
        torch.empty(1, dtype=dtype),  # guidance_scale
        torch.empty(1, dtype=torch.int64),  # timestep
    ]

    @fxb.export_program(
        args=(example_init_args,),
    )
    def _initialize(module, sample):
        return module.initialize(*sample)

    @fxb.export_program(
        args=(example_forward_args,),
    )
    def _forward(
        module,
        inputs,
    ):
        return module.forward(*inputs)

    decomp_list = []
    if decomp_attn == True:
        decomp_list.extend(
            [
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
                torch.ops.aten._scaled_dot_product_flash_attention.default,
            ]
        )
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):

        class CompiledScheduledUnet(CompiledModule):
            run_initialize = _initialize
            run_forward = _forward

        if external_weights:
            externalize_module_parameters(scheduled_unet_model)
        if external_weight_path and len(external_weight_path) > 1:
            save_module_parameters(external_weight_path, scheduled_unet_model)

        inst = CompiledScheduledUnet(context=Context(), import_to="IMPORT")

        module_str = str(CompiledModule.get_mlir_module(inst))

    if compile_to != "vmfb":
        return module_str
    elif compile_to == "vmfb":
        vmfb = utils.compile_to_vmfb(
            module_str,
            device,
            iree_target_triple,
            ireec_flags,
            safe_name,
            return_path=True,
            attn_spec=attn_spec,
        )
        if exit_on_vmfb:
            exit()
        return vmfb


def export_pipeline_module(args):
    from turbine_models.custom_models.sdxl_inference.pipeline_ir import get_pipeline_ir

    pipeline_file = get_pipeline_ir(
        args.width,
        args.height,
        args.precision,
        args.batch_size,
        args.max_length,
        "unet_loop",
    )
    pipeline_vmfb = utils.compile_to_vmfb(
        pipeline_file,
        args.device,
        args.iree_target_triple,
        None,
        os.path.join(args.pipeline_dir, "pipeline"),
        return_path=True,
        mlir_source="str",
    )
    full_pipeline_file = get_pipeline_ir(
        args.width,
        args.height,
        args.precision,
        args.batch_size,
        args.max_length,
        "tokens_to_image",
    )
    full_pipeline_vmfb = utils.compile_to_vmfb(
        pipeline_file,
        args.device,
        args.iree_target_triple,
        None,
        os.path.join(args.pipeline_dir, "pipeline"),
        return_path=True,
        mlir_source="str",
    )
    return full_pipeline_vmfb


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    if args.input_mlir:
        scheduled_unet_model = None
    else:
        scheduled_unet_model = SDXLScheduledUnet(
            args.hf_model_name,
            args.scheduler_id,
            args.height,
            args.width,
            args.batch_size,
            args.hf_auth_token,
            args.precision,
            args.num_inference_steps,
            args.return_index,
        )
    if args.compile_to == "vmfb" and args.pipeline_dir is not None:
        pipeline_vmfb_path = export_pipeline_module(args)
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
        args.ireec_flags + args.attn_flags + args.unet_flags,
        args.decomp_attn,
        args.exit_on_vmfb,
        args.pipeline_dir,
        args.attn_spec,
        args.input_mlir,
    )
    if args.input_mlir:
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name + "_" + args.scheduler_id,
        f"_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}_unet_{str(args.num_inference_steps)}",
    )
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
