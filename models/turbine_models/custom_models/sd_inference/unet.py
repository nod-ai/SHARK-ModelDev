# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import copy

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

import safetensors
import argparse
from turbine_models.turbine_tank import turbine_tank


class UnetModel(torch.nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            hf_model_name,
            subfolder="unet",
        )

    def forward(self, sample, timestep, encoder_hidden_states, guidance_scale):
        samples = torch.cat([sample] * 2)
        unet_out = self.unet.forward(
            samples, timestep, encoder_hidden_states, return_dict=False
        )[0]
        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred


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
    if "turbo" in hf_model_name:
        do_classifier_free_guidance = False
    else:
        do_classifier_free_guidance = True
    if pipeline_dir:
        safe_name = os.path.join(pipeline_dir, f"unet")
    else:
        safe_name = utils.create_safe_name(
            hf_model_name,
            f"_bs{batch_size}_{max_length}_{height}x{width}_{precision}_unet_{device}",
        )
    if input_mlir:
        vmfb_path = utils.compile_to_vmfb(
            input_mlir,
            device,
            target_triple,
            ireec_flags,
            safe_name,
            mlir_source="file",
            return_path=not exit_on_vmfb,
            attn_spec=attn_spec,
        )
        return vmfb_path

    mapper = {}
    decomp_list = copy.deepcopy(DEFAULT_DECOMPOSITIONS)
    if decomp_attn == True:
        decomp_list.extend(
            [
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
                torch.ops.aten._scaled_dot_product_flash_attention.default,
            ]
        )
    dtype = torch.float16 if precision == "fp16" else torch.float32

    if precision == "fp16":
        unet_model = unet_model.half()

    utils.save_external_weights(
        mapper, unet_model, external_weights, external_weight_path
    )

    if weights_only:
        return external_weight_path

    sample = (
        batch_size,
        unet_model.unet.config.in_channels,
        height // 8,
        width // 8,
    )

    encoder_hidden_states_sizes = (
        unet_model.unet.config.layers_per_block,
        max_length,
        unet_model.unet.config.cross_attention_dim,
    )

    class CompiledUnet(CompiledModule):
        if external_weights:
            params = export_parameters(
                unet_model, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(unet_model)

        def main(
            self,
            sample=AbstractTensor(*sample, dtype=dtype),
            timestep=AbstractTensor(1, dtype=dtype),
            encoder_hidden_states=AbstractTensor(
                *encoder_hidden_states_sizes, dtype=dtype
            ),
            guidance_scale=AbstractTensor(1, dtype=dtype),
        ):
            return jittable(unet_model.forward, decompose_ops=decomp_list)(
                sample, timestep, encoder_hidden_states, guidance_scale
            )

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledUnet(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))

    if compile_to != "vmfb":
        return module_str
    else:
        utils.compile_to_vmfb(
            module_str,
            device,
            target_triple,
            ireec_flags,
            safe_name,
            return_path=False,
            attn_spec=attn_spec,
        )


if __name__ == "__main__":
    from turbine_models.custom_models.sd_inference.sd_cmd_opts import args

    if args.input_mlir:
        unet_model = None
    else:
        unet_model = UnetModel(
            args.hf_model_name,
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
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
