# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import os
import sys
import math

from safetensors import safe_open
from iree import runtime as ireert
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from shark_turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
from shark_turbine.transforms.general.add_metadata import AddMetadataPass
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft
from flux.sampling import unpack
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.util import configs, print_load_warning
from einops import rearrange


# The following model loader is derived from https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py#L138
def load_ae(
    name: str, device: str | torch.device = "cpu", hf_download: bool = True
) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae


class AEModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name="flux-dev",
        height=1024,
        width=1024,
        dtype=torch.float16,
    ):
        super().__init__()
        self.ae = load_ae(hf_model_name).to(dtype)
        self.height = height
        self.width = width
        self.dtype = dtype

    def decode(
        self,
        img,
    ):
        latents = unpack(img, self.height, self.width)
        image = self.ae.decode(latents)
        image = image.clamp(-1, 1)
        image = rearrange(image[0], "c h w -> h w c")
        return 127.5 * (image + 1.0)


@torch.no_grad()
def export_ae_model(
    hf_model_name,
    batch_size,
    height,
    width,
    num_channels=16,
    precision="fp16",
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target=None,
    ireec_flags="",
    decomp_attn=False,
    exit_on_vmfb=False,
    pipeline_dir=None,
    attn_spec=None,
    input_mlir=None,
    weights_only=False,
):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    np_dtype = "float16" if precision == "fp16" else "float32"
    safe_name = utils.create_safe_name(
        hf_model_name,
        f"_bs{batch_size}_{height}x{width}_{precision}_ae",
    )
    if pipeline_dir:
        safe_name = os.path.join(pipeline_dir, safe_name)
    if decomp_attn == True:
        safe_name += "_decomp_attn"
        ireec_flags += ",--iree-opt-aggressively-propagate-transposes=False"

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

    ae_model = AEModel(hf_model_name, height, width, dtype)

    mapper = {}

    utils.save_external_weights(
        mapper, ae_model, external_weights, external_weight_path
    )

    if weights_only:
        return external_weight_path

    img_shape = (
        batch_size,
        int(height * width / 256),
        64,
    )

    example_forward_args = [
        torch.empty(img_shape, dtype=dtype),
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
        fxb = FxProgramsBuilder(ae_model)

        @fxb.export_program(
            args=(example_forward_args,),
        )
        def _decode(
            module,
            inputs,
        ):
            return module.decode(*inputs)

        class CompiledFluxAE(CompiledModule):
            decode = _decode

        if external_weights:
            externalize_module_parameters(ae_model)

        inst = CompiledFluxAE(context=Context(), import_to="IMPORT")

        module = CompiledModule.get_mlir_module(inst)

    model_metadata_decode = {
        "model_name": "flux_ae",
        "input_shapes": [
            img_shape,
        ],
        "input_dtypes": [np_dtype],
        "output_dtypes": [np_dtype],
    }
    module = AddMetadataPass(module, model_metadata_decode, "decode").run()
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
    import logging

    logging.basicConfig(level=logging.DEBUG)
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args

    mod_str = export_ae_model(
        args.hf_model_name,
        args.batch_size,
        args.height,
        args.width,
        args.precision,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.ireec_flags,
        args.decomp_attn,
        attn_spec=args.attn_spec,
        input_mlir=args.input_mlir,
        weights_only=args.weights_only,
    )
    if args.input_mlir:
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name,
        f"_bs{args.batch_size}_{args.height}x{args.width}_{args.precision}_sampler",
    )
    if args.compile_to != "vmfb":
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
