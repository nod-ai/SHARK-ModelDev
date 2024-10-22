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
from iree.turbine.aot import *
from iree.turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
from iree.turbine.transforms.general.add_metadata import AddMetadataPass
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
        self.do_classifier_free_guidance = True
        self.unet = UNet2DConditionModel.from_pretrained(
            hf_model_name,
            subfolder="unet",
        )

    def forward(
        self, latent_model_input, timestep, encoder_hidden_states, guidance_scale
    ):
        noise_pred = self.unet.forward(
            latent_model_input, timestep, encoder_hidden_states, return_dict=False
        )[0]
        if self.do_classifier_free_guidance:
            noise_preds = noise_pred.chunk(2)
            noise_pred = noise_preds[0] + guidance_scale * (
                noise_preds[1] - noise_preds[0]
            )
        return noise_pred


def export_unet_model(
    hf_model_name,
    batch_size,
    height,
    width,
    precision="fp32",
    max_length=77,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target=None,
    ireec_flags=None,
    decomp_attn=False,
    exit_on_vmfb=False,
    pipeline_dir=None,
    attn_spec=None,
    input_mlir=None,
    weights_only=False,
    upload_ir=False,
):
    if input_mlir:
        unet_model = None
    else:
        unet_model = UnetModel(
            hf_model_name,
        )
    dtype = torch.float16 if precision == "fp16" else torch.float32
    np_dtype = "float16" if precision == "fp16" else "float32"
    safe_name = utils.create_safe_name(
        hf_model_name,
        f"_bs{batch_size}_{max_length}_{height}x{width}_{precision}_unet",
    )
    if decomp_attn:
        safe_name += "_decomp_attn"
    if pipeline_dir:
        safe_name = os.path.join(pipeline_dir, safe_name)

    if input_mlir:
        vmfb_path = utils.compile_to_vmfb(
            input_mlir,
            device,
            target,
            ireec_flags,
            safe_name,
            mlir_source="file",
            return_path=not exit_on_vmfb,
            attn_spec=attn_spec,
        )
        return vmfb_path

    mapper = {}

    if precision == "fp16":
        unet_model = unet_model.half()

    utils.save_external_weights(
        mapper, unet_model, external_weights, external_weight_path
    )

    if weights_only:
        return external_weight_path

    sample = (
        batch_size * 2,
        unet_model.unet.config.in_channels,
        height // 8,
        width // 8,
    )
    encoder_hidden_states_sizes = (
        unet_model.unet.config.layers_per_block,
        max_length,
        unet_model.unet.config.cross_attention_dim,
    )
    example_forward_args = [
        torch.empty(sample, dtype=dtype),
        torch.empty(1, dtype=dtype),
        torch.empty(encoder_hidden_states_sizes, dtype=dtype),
        torch.empty(1, dtype=dtype),
    ]
    decomp_list = []
    if decomp_attn:
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

        module = CompiledModule.get_mlir_module(inst)

    model_metadata_run_forward = {
        "model_name": "sd_unet",
        "input_shapes": [
            sample,
            (1,),
            encoder_hidden_states_sizes,
            (1,),
        ],
        "input_dtypes": [np_dtype for x in range(4)],
        "output_shapes": [sample],
        "output_dtypes": [np_dtype],
    }

    module = AddMetadataPass(module, model_metadata_run_forward, "run_forward").run()
    module_str = str(module)
    if compile_to != "vmfb":
        return module_str
    else:
        vmfb_path = utils.compile_to_vmfb(
            module_str,
            device,
            target,
            ireec_flags,
            safe_name,
            return_path=True,
            attn_spec=attn_spec,
        )
        if exit_on_vmfb:
            exit()
    return vmfb_path


if __name__ == "__main__":
    from turbine_models.custom_models.sd_inference.sd_cmd_opts import args

    mod_str = export_unet_model(
        args.hf_model_name,
        args.batch_size,
        args.height,
        args.width,
        args.precision,
        args.max_length,
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
