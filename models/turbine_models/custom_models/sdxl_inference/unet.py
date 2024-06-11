# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import os
import sys

from iree import runtime as ireert
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from shark_turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from diffusers import UNet2DConditionModel


class UnetModel(torch.nn.Module):
    def __init__(self, hf_model_name, hf_auth_token=None, precision="fp32"):
        super().__init__()
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
        # if "turbo" in hf_model_name:
        #     self.do_classifier_free_guidance = False
        # else:
        self.do_classifier_free_guidance = True

    def forward(
        self, latent_model_input, timestep, prompt_embeds, text_embeds, time_ids
    ):
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        noise_pred = self.unet.forward(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return noise_pred


@torch.no_grad()
def export_unet_model(
    unet_model,
    hf_model_name,
    batch_size,
    height,
    width,
    precision="fp32",
    max_length=77,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target_triple=None,
    ireec_flags=None,
    decomp_attn=False,
    exit_on_vmfb=False,
    pipeline_dir=None,
    attn_spec=None,
    input_mlir=None,
    weights_only=False,
):
    if pipeline_dir:
        safe_name = os.path.join(
            pipeline_dir, f"unet"
        )
    else:
        safe_name = utils.create_safe_name(
            hf_model_name,
            f"_bs{batch_size}_{max_length}_{height}x{width}_{precision}_unet",
        )
    if decomp_attn == True:
        ireec_flags += ",--iree-opt-aggressively-propagate-transposes=False"

    if input_mlir:
        vmfb_path = utils.compile_to_vmfb(
            input_mlir,
            device,
            target_triple,
            ireec_flags,
            safe_name + "_" + target_triple,
            mlir_source="file",
            return_path=not exit_on_vmfb,
            attn_spec=attn_spec,
        )
        return vmfb_path

    mapper = {}
    dtype = torch.float16 if precision == "fp16" else torch.float32

    if precision == "fp16":
        unet_model = unet_model.half()

    utils.save_external_weights(
        mapper, unet_model, external_weights, external_weight_path
    )

    if weights_only:
        return external_weight_path

    do_classifier_free_guidance = True
    init_batch_dim = 2 if do_classifier_free_guidance else 1

    prepared_latents = (
        batch_size * init_batch_dim,
        unet_model.unet.config.in_channels,
        height // 8,
        width // 8,
    )

    time_ids_shape = (init_batch_dim * batch_size, 6)
    prompt_embeds_shape = (init_batch_dim * batch_size, max_length, 2048)
    text_embeds_shape = (init_batch_dim * batch_size, 1280)
    example_forward_args = [
        torch.empty(prepared_latents, dtype=dtype),
        torch.empty(1, dtype=dtype),
        torch.empty(prompt_embeds_shape, dtype=dtype),
        torch.empty(text_embeds_shape, dtype=dtype),
        torch.empty(time_ids_shape, dtype=dtype),
    ]

    decomp_list = []
    if decomp_attn == True:
        decomp_list = [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ]
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):
        fxb = FxProgramsBuilder(unet_model)

        @fxb.export_program(
            args=(example_forward_args,),
        )
        def _forward(
            module,
            inputs,
        ):
            return module.forward(*inputs)

        class CompiledUnet(CompiledModule):
            run_forward = _forward

        if external_weights:
            externalize_module_parameters(unet_model)

        inst = CompiledUnet(context=Context(), import_to="IMPORT")

        module_str = str(CompiledModule.get_mlir_module(inst))

    if compile_to != "vmfb":
        return module_str
    else:
        vmfb_path = utils.compile_to_vmfb(
            module_str,
            device,
            target_triple,
            ireec_flags,
            safe_name,
            return_path=True,
            attn_spec=attn_spec,
        )
        if exit_on_vmfb:
            exit()
    return vmfb_path


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    if args.input_mlir:
        unet_model = None
    else:
        unet_model = UnetModel(
            args.hf_model_name,
            args.hf_auth_token,
            args.precision,
        )
    mod_str = export_unet_model(
        unet_model,
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
        attn_spec=args.attn_spec,
        input_mlir=args.input_mlir,
    )
    if args.input_mlir:
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name,
        f"_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}_unet",
    )
    if args.compile_to != "vmfb":
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
