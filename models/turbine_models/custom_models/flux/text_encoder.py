# Copyright 2024 Advanced Micro Devices, inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

import math
import safetensors
import iree.compiler as ireec
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from shark_turbine.transforms.general.add_metadata import AddMetadataPass
from turbine_models.custom_models.sd_inference import utils
from transformers import T5EncoderModel
from einops import rearrange, repeat
import torch
from turbine_models.custom_models.sd3_inference.text_encoder_impls import (
    SDClipModel,
    load_into,
)
from huggingface_hub import hf_hub_download
from safetensors import safe_open


CLIP_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3172,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

model_repo_map = {
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux-pro": "black-forest-labs/FLUX.1-pro",
}


class TextEncoderModule(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        hf_model_name,
        precision,
        batch_size=1,
        max_length=64,
    ):
        super().__init__()
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.clip = SDClipModel(
            layer="hidden",
            layer_idx=-2,
            device="cpu",
            dtype=self.dtype,
            layer_norm_hidden_state=False,
            return_projected_pooled=False,
            textmodel_json_config=CLIP_CONFIG,
        )
        if precision == "fp16":
            self.clip = self.clip.half()
        clip_weights = hf_hub_download(
            repo_id=model_repo_map[hf_model_name],
            filename="text_encoder/model.safetensors",
        )
        with safe_open(clip_weights, framework="pt", device="cpu") as f:
            load_into(f, self.clip.transformer, "", "cpu", self.dtype)

        self.t5 = T5EncoderModel.from_pretrained(
            model_repo_map[hf_model_name],
            max_length=max_length,
            torch_dtype=torch.float16,
            subfolder="text_encoder_2",
        )
        self.batch_size = batch_size
        self.max_length = max_length

    def forward(self, t5_ids, clip_ids):
        txt = self.t5(t5_ids)
        clip_out, vec = self.clip(clip_ids)

        return txt, vec


@torch.no_grad()
def export_text_encoders(
    hf_model_name="flux-dev",
    max_length=64,
    batch_size=1,
    precision="fp16",
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target=None,
    ireec_flags=None,
    exit_on_vmfb=False,
    pipeline_dir=None,
    input_mlir=None,
    attn_spec=None,
    decomp_attn=False,
    weights_only=False,
):
    safe_name = utils.create_safe_name(
        hf_model_name,
        f"_bs{batch_size}_{max_length}_{precision}_text_encoders",
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
            const_expr_hoisting=True,
            attn_spec=attn_spec,
        )
        return vmfb_path
    model = TextEncoderModule(
        hf_model_name,
        precision,
        batch_size,
        max_length,
    )
    if precision == "fp16":
        model = model.half()
    mapper = {}

    assert (
        ".safetensors" not in external_weight_path
    ), "Original parameters format incompatible with IREE safetensors parser. Use '.irpa' instead."

    t5_ids_shape = (
        batch_size,
        max_length,
    )
    clip_ids_shape = (
        batch_size,
        77,
        2,
    )
    input_args = [
        torch.ones(t5_ids_shape, dtype=torch.int64),
        torch.ones(clip_ids_shape, dtype=torch.int64),
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
        fxb = FxProgramsBuilder(model)

        @fxb.export_program(
            args=(input_args,),
        )
        def _forward(
            module,
            inputs,
        ):
            return module.forward(*inputs)

        class CompiledTextEncoder(CompiledModule):
            encode_tokens = _forward

        if external_weights:
            externalize_module_parameters(model)
            save_module_parameters(external_weight_path, model)

        inst = CompiledTextEncoder(context=Context(), import_to="IMPORT")

        module = CompiledModule.get_mlir_module(inst)

    # model_metadata_forward = {
    #     "model_name": "flux_clip_t5xxl_text_encoders",
    #     # "input_shapes": [(batch_size, max_length, 2) for x in range(6)],
    #     # "input_dtypes": ["int64" for x in range(6)],
    #     # "output_shapes": [
    #     #     (2 * batch_size, max_length * 2, 4096),
    #     #     (2 * batch_size, 2048),
    #     # ],
    #     # "output_dtypes": ["float32"],
    # }
    # module = AddMetadataPass(module, model_metadata_forward, "forward").run()
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
            return_path=not exit_on_vmfb,
            const_expr_hoisting=True,
            attn_spec=attn_spec,
        )
        return vmfb_path


if __name__ == "__main__":
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args

    mod_str = export_text_encoders(
        "flux-dev",
        args.max_length,
        args.batch_size,
        args.precision,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.ireec_flags + args.clip_flags,
        exit_on_vmfb=True,
        pipeline_dir=args.pipeline_dir,
        input_mlir=args.input_mlir,
        attn_spec=args.attn_spec,
    )
    if args.input_mlir or args.weights_only or args.compile_to == "vmfb":
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name, f"_{str(args.max_length)}_{args.precision}_text_encoders"
    )
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
