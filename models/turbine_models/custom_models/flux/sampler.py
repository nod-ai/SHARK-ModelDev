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
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.model import Flux, FluxParams
from flux.util import configs, print_load_warning
from flux.modules.layers import DoubleStreamBlock

# The following model loader is derived from https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py#L105
def load_flux_model(name: str, device = "cpu", hf_download:bool = True):
    # Loading Flux
    print("Init Flux sampling model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params).to(torch.float16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return model


class FluxModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name="flux-dev",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.sampler = load_flux_model(hf_model_name)
        self.dtype = dtype

    def forward(
        self,
        img,
        img_ids,
        txt,
        txt_ids,
        vec,
        t_curr,
        t_prev,
        guidance_vec,
    ):
        t_vec = t_curr.expand(img.shape[0])
        noise_pred = self.sampler(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        img = img + (t_prev - t_curr) * noise_pred
        return img

@torch.no_grad()
def export_flux_model(
    hf_model_name,
    batch_size,
    height,
    width,
    precision="fp16",
    max_length=256,
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
    dtype = torch.float16
    np_dtype = "float16"
    safe_name = utils.create_safe_name(
        hf_model_name,
        f"_bs{batch_size}_{height}x{width}_{precision}_sampler",
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

    flux_model = FluxModel(
        hf_model_name, dtype=dtype,
    ).half()
    mapper = {}

    utils.save_external_weights(
        mapper, flux_model, external_weights, external_weight_path
    )

    if weights_only:
        return external_weight_path
    model_max_len = 256 if "schnell" in hf_model_name else 512

    img_shape = (
        batch_size,
        int(height * width / 256),
        64,
    )
    img_ids_shape = (
        batch_size,
        int(height * width / 256),
        3,
    )
    txt_shape = (
        batch_size,
        model_max_len,
        4096,
    )
    txt_ids_shape = (
        batch_size,
        model_max_len,
        3,
    )
    y_shape = (
        batch_size,
        768,
    )
    example_forward_args = [
        torch.empty(img_shape, dtype=dtype),
        torch.empty(img_ids_shape, dtype=dtype),
        torch.empty(txt_shape, dtype=dtype),
        torch.empty(txt_ids_shape, dtype=dtype),
        torch.empty(y_shape, dtype=dtype),
        torch.empty(1, dtype=dtype),
        torch.empty(1, dtype=dtype),
        torch.empty(1, dtype=dtype),
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
        fxb = FxProgramsBuilder(flux_model)

        @fxb.export_program(
            args=(example_forward_args,),
        )
        def _forward(
            module,
            inputs,
        ):
            return module.forward(*inputs)

        class CompiledFluxSampler(CompiledModule):
            run_forward = _forward

        if external_weights:
            externalize_module_parameters(flux_model)

        inst = CompiledFluxSampler(context=Context(), import_to="IMPORT")

        module = CompiledModule.get_mlir_module(inst)

    model_metadata_run_forward = {
        "model_name": "flux_sampler",
        # "input_shapes": [
        #     hidden_states_shape,
        #     encoder_hidden_states_shape,
        #     pooled_projections_shape,
        #     (1,),
        # ],
        # "input_dtypes": [np_dtype for x in range(4)],
        # "output_shapes": [hidden_states_shape],
        # "output_dtypes": [np_dtype],
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

class FluxAttention(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.attn = DoubleStreamBlock(
            3072,
            24,
            mlp_ratio=4.0,
            qkv_bias=True,
        )

    def forward(self, img, txt, vec, pe):
        return self.attn.forward(img=img, txt=txt, vec=vec, pe=pe)


@torch.no_grad()
def export_attn(
    precision="fp16",
    device="cpu",
    target="x86_64-unknown-linux-gnu",
    ireec_flags="",
    compile_to="torch",
    decomp_attn=False,
    attn_spec=None,
):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    attn_module = FluxAttention()
    safe_name = "flux_sampler_attn_repro_" + precision
    if decomp_attn == True:
        safe_name += "_decomp"

    if dtype == torch.float16:
        attn_module = attn_module.half()

    example_args = [
        torch.empty((1, 4096, 3072), dtype=dtype),
        torch.empty((1, 512, 3072), dtype=dtype),
        torch.empty((1, 3072), dtype=dtype),
        torch.empty((1, 1, 4608, 64, 2, 2), dtype=torch.float32),
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
        fxb = FxProgramsBuilder(attn_module)

        @fxb.export_program(
            args=(example_args,),
        )
        def _forward(
            module,
            inputs,
        ):
            return module.forward(*inputs)

        class CompiledAttn(CompiledModule):
            main = _forward

        externalize_module_parameters(attn_module)

        inst = CompiledAttn(context=Context(), import_to="IMPORT")

        module_str = str(CompiledModule.get_mlir_module(inst))

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
    return vmfb_path

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args

    if args.attn_repro:
        mod_str = export_attn(
            args.precision,
            args.device,
            args.iree_target_triple,
            args.ireec_flags,
            args.compile_to,
            args.decomp_attn,
            attn_spec=args.attn_spec,
        )
        if args.compile_to != "vmfb":
            safe_name = "flux_attn_repro_" + args.precision
            with open(f"{safe_name}.mlir", "w+") as f:
                f.write(mod_str)
            print("Saved to", safe_name + ".mlir")
        exit()
    
    mod_str = export_flux_model(
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
        attn_spec=args.attn_spec,
        input_mlir=args.input_mlir,
        weights_only=args.weights_only,
    )
    if args.input_mlir:
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name,
        f"_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}_sampler",
    )
    if args.compile_to != "vmfb":
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
