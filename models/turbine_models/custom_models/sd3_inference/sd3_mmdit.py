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
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from diffusers import SD3Transformer2DModel


class MMDiTModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name="stabilityai/stable-diffusion-3-medium-diffusers",
        dtype=torch.float16,
    ):
        super().__init__()
        self.mmdit = SD3Transformer2DModel.from_pretrained(
            hf_model_name,
            subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
    ):
        noise_pred = self.mmdit(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            return_dict=False,
        )[0]
        return noise_pred


class MMDiTAttention(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )


@torch.no_grad()
def export_attn(
    precision="fp16",
    device="cpu",
    target_triple="x86_64-unknown-linux-gnu",
    ireec_flags="",
    compile_to="torch",
    decomp_attn=False,
    attn_spec=None,
):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    qkv_shape = (2, 24, 4250, 64)
    attn_module = MMDiTAttention()
    safe_name = "attn_repro_" + precision + "_" + target_triple
    if decomp_attn == True:
        safe_name += "_decomp"

    if dtype == torch.float16:
        attn_module = attn_module.half()

    example_qkv = [
        torch.empty(qkv_shape, dtype=dtype),
        torch.empty(qkv_shape, dtype=dtype),
        torch.empty(qkv_shape, dtype=dtype),
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
            args=(example_qkv,),
        )
        def _forward(
            module,
            inputs,
        ):
            return module.forward(*inputs)

        class CompiledAttn(CompiledModule):
            run_forward = _forward

        inst = CompiledAttn(context=Context(), import_to="IMPORT")

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
    return vmfb_path


@torch.no_grad()
def export_mmdit_model(
    mmdit_model,
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
    dtype = torch.float16 if precision == "fp16" else torch.float32
    if pipeline_dir:
        safe_name = os.path.join(pipeline_dir, f"mmdit")
    else:
        safe_name = utils.create_safe_name(
            hf_model_name,
            f"_bs{batch_size}_{max_length}_{height}x{width}_{precision}_mmdit",
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

    utils.save_external_weights(
        mapper, mmdit_model, external_weights, external_weight_path
    )

    if weights_only:
        return external_weight_path

    do_classifier_free_guidance = True
    init_batch_dim = 2 if do_classifier_free_guidance else 1
    batch_size = batch_size * init_batch_dim
    hidden_states_shape = (
        batch_size,
        16,
        height // 8,
        width // 8,
    )
    encoder_hidden_states_shape = (batch_size, 154, 4096)
    pooled_projections_shape = (batch_size, 2048)
    example_forward_args = [
        torch.empty(hidden_states_shape, dtype=dtype),
        torch.empty(encoder_hidden_states_shape, dtype=dtype),
        torch.empty(pooled_projections_shape, dtype=dtype),
        torch.empty(init_batch_dim, dtype=dtype),
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
        fxb = FxProgramsBuilder(mmdit_model)

        @fxb.export_program(
            args=(example_forward_args,),
        )
        def _forward(
            module,
            inputs,
        ):
            return module.forward(*inputs)

        class CompiledMmdit(CompiledModule):
            run_forward = _forward

        if external_weights:
            externalize_module_parameters(mmdit_model)

        inst = CompiledMmdit(context=Context(), import_to="IMPORT")

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
            safe_name = "attn_repro_" + args.precision
            with open(f"{safe_name}.mlir", "w+") as f:
                f.write(mod_str)
            print("Saved to", safe_name + ".mlir")
        exit()
    if args.input_mlir:
        mmdit_model = None
    else:
        mmdit_model = MMDiTModel(
            args.hf_model_name,
            dtype=torch.float16 if args.precision == "fp16" else torch.float32,
        )
    mod_str = export_mmdit_model(
        mmdit_model,
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
        weights_only=args.weights_only,
    )
    if args.input_mlir:
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name,
        f"_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}_mmdit",
    )
    if args.compile_to != "vmfb":
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
