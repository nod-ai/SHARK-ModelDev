# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

import safetensors
from iree import runtime as ireert
import iree.compiler as ireec
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from shark_turbine.ops.iree import trace_tensor
from shark_turbine.transforms.general.add_metadata import AddMetadataPass
from turbine_models.custom_models.sd_inference import utils
import torch
from turbine_models.custom_models.sd3_inference.text_encoder_impls import (
    SDClipModel,
    SDXLClipG,
    T5XXLModel,
    load_into,
)
from huggingface_hub import hf_hub_download
from safetensors import safe_open

CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class TextEncoderModule(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
    ):
        super().__init__()
        self.dtype = torch.float16
        self.clip_l = SDClipModel(
            layer="hidden",
            layer_idx=-2,
            device="cpu",
            dtype=self.dtype,
            layer_norm_hidden_state=False,
            return_projected_pooled=False,
            textmodel_json_config=CLIPL_CONFIG,
        ).half()
        clip_l_weights = hf_hub_download(
            repo_id="stabilityai/stable-diffusion-3-medium",
            filename="text_encoders/clip_l.safetensors",
        )
        with safe_open(clip_l_weights, framework="pt", device="cpu") as f:
            load_into(f, self.clip_l.transformer, "", "cpu", self.dtype)
        self.clip_g = SDXLClipG(CLIPG_CONFIG, device="cpu", dtype=self.dtype).half()
        clip_g_weights = hf_hub_download(
            repo_id="stabilityai/stable-diffusion-3-medium",
            filename="text_encoders/clip_g.safetensors",
        )
        with safe_open(clip_g_weights, framework="pt", device="cpu") as f:
            load_into(f, self.clip_g.transformer, "", "cpu", self.dtype)
        self.t5xxl = T5XXLModel(T5_CONFIG, device="cpu", dtype=self.dtype).half()
        t5_weights = hf_hub_download(
            repo_id="stabilityai/stable-diffusion-3-medium",
            filename="text_encoders/t5xxl_fp16.safetensors",
        )
        with safe_open(t5_weights, framework="pt", device="cpu") as f:
            load_into(f, self.t5xxl.transformer, "", "cpu", self.dtype)

        self.do_classifier_free_guidance = True

    def get_cond(self, tokens_l, tokens_g, tokens_t5xxl):
        l_out, l_pooled = self.clip_l.forward(tokens_l)
        g_out, g_pooled = self.clip_g.forward(tokens_g)
        t5_out, _ = self.t5xxl.forward(tokens_t5xxl)
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )

    def forward(self, tokens_g, tokens_l, tokens_t5xxl, neg_g, neg_l, neg_t5):
        conditioning, cond_pool = self.get_cond(tokens_l, tokens_g, tokens_t5xxl)
        neg_cond, neg_cond_pool = self.get_cond(neg_l, neg_g, neg_t5)

        prompt_embeds = torch.cat([neg_cond, conditioning], dim=0)
        pooled_prompt_embeds = torch.cat([neg_cond_pool, cond_pool], dim=0)

        return prompt_embeds, pooled_prompt_embeds


@torch.no_grad()
def export_text_encoders(
    hf_model_name,
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
    decomp_attn=True,
):

    safe_name = utils.create_safe_name(
        hf_model_name,
        f"_bs{batch_size}_{str(max_length)}_{precision}_text_encoders",
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
    model = TextEncoderModule()

    assert (
        ".safetensors" not in external_weight_path
    ), "Original parameters format incompatible with IREE safetensors parser. Use '.irpa' instead."

    input_args = [torch.empty([batch_size, 77, 2], dtype=torch.int64) for x in range(6)]

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

    model_metadata_forward = {
        "model_name": "sd3_clip_t5xxl_text_encoders",
        "input_shapes": [(batch_size, max_length, 2) for x in range(6)],
        "input_dtypes": ["int64" for x in range(6)],
        "output_shapes": [
            (2 * batch_size, max_length * 2, 4096),
            (2 * batch_size, 2048),
        ],
        "output_dtypes": ["float32"],
    }
    module = AddMetadataPass(module, model_metadata_forward, "forward").run()
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

    mod_str, _ = export_text_encoders(
        args.hf_model_name,
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
