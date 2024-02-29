# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree import runtime as ireert
import iree.compiler as ireec
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token, required"
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="stabilityai/stable-diffusion-xl-base-1.0",
)
parser.add_argument("--max_length", type=int, default=77)
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument(
    "--precision", type=str, default="fp16", help="Precision of Stable Diffusion"
)
parser.add_argument("--external_weight_path", type=str, default="")
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="saves ir/vmfb without global weights for size and readability, options [safetensors]",
)
parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, vulkan, rocm")
# TODO: Bring in detection for target triple
parser.add_argument(
    "--iree_target_triple",
    type=str,
    default="",
    help="Specify vulkan target triple or rocm/cuda target device.",
)
parser.add_argument("--vulkan_max_allocation", type=str, default="4294967296")

class ClipModel(torch.nn.Module):
    def __init__(self, hf_model_name, hf_auth_token=None, index=1):
        super().__init__()
        if index == 1:
            self.text_encoder_model = CLIPTextModel.from_pretrained(
                hf_model_name,
                subfolder="text_encoder",
                token=hf_auth_token,
            )
        if index == 2:
            self.text_encoder_model = CLIPTextModelWithProjection.from_pretrained(
                hf_model_name,
                subfolder="text_encoder_2",
                token=hf_auth_token,
            )

    def forward(self, input):
        with torch.no_grad():           
            prompt_embeds = self.text_encoder_model(
                input,
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
        return prompt_embeds, pooled_prompt_embeds

def export_clip_model(
    hf_model_name,
    hf_auth_token=None,
    max_length=77,
    precision="fp16",
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target_triple=None,
    index=1,
    max_alloc=None,
    exit_on_vmfb=True,
):
    # Load the tokenizer and text encoder to tokenize and encode the text.
    if index == 1:
        tokenizer = CLIPTokenizer.from_pretrained(
            hf_model_name,
            subfolder="tokenizer",
            token=hf_auth_token,
        )
    elif index == 2:
        tokenizer = CLIPTokenizer.from_pretrained(
            hf_model_name,
            subfolder="tokenizer_2",
            token=hf_auth_token,
        )
    text_encoder_model = ClipModel(hf_model_name, hf_auth_token, index=index)
    if compile_to == "tokenizer_only":
        return None, tokenizer
    if precision == "fp16":
        text_encoder_model = text_encoder_model.half()
    mapper = {}
    if external_weight_path:
        weights_path = (
            external_weight_path.split(f".{external_weights}")[0]
            + f"_{index}"
            + f".{external_weights}"
        )
    else:
        weights_path = None

    utils.save_external_weights(
        mapper, text_encoder_model, external_weights, weights_path
    )

    class CompiledClip(CompiledModule):
        if external_weights:
            params = export_parameters(
                text_encoder_model,
                external=True,
                external_scope="",
                name_mapper=mapper.get,
            )
        else:
            params = export_parameters(text_encoder_model)

        def main(self, inp=AbstractTensor(1, max_length, dtype=torch.int64)):
            return jittable(text_encoder_model.forward)(inp)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledClip(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = utils.create_safe_name(
        hf_model_name, f"-{str(max_length)}-{precision}-clip-{index}-{device}"
    )
    if compile_to != "vmfb":
        return module_str, tokenizer
    elif os.path.isfile(safe_name + ".vmfb"):
        exit()
    elif exit_on_vmfb == False:
        vmfb_path = utils.compile_to_vmfb(
            module_str,
            device,
            target_triple,
            max_alloc,
            safe_name,
            return_path=True,
            const_eval=True,
        )
        return None, vmfb_path
    else:
        utils.compile_to_vmfb(
            module_str,
            device,
            target_triple,
            max_alloc,
            safe_name,
            const_expr_hoisting=True,
        )


if __name__ == "__main__":
    import re

    args = parser.parse_args()
    mod_1_str, _ = export_clip_model(
        args.hf_model_name,
        args.hf_auth_token,
        args.max_length,
        args.precision,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        1,
        args.vulkan_max_allocation,
        exit_on_vmfb=False,
    )
    mod_2_str, _ = export_clip_model(
        args.hf_model_name,
        args.hf_auth_token,
        args.max_length,
        args.precision,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        2,
        args.vulkan_max_allocation,
        exit_on_vmfb=True,
    )
    safe_name_1 = safe_name = utils.create_safe_name(
        args.hf_model_name, f"_{str(args.max_length)}_{args.precision}_clip_1"
    )
    safe_name_2 = safe_name = utils.create_safe_name(
        args.hf_model_name, f"_{str(args.max_length)}_{args.precision}_clip_2"
    )
    with open(f"{safe_name_1}.mlir", "w+") as f:
        f.write(mod_1_str)
    print("Saved to", safe_name_1 + ".mlir")
    with open(f"{safe_name_2}.mlir", "w+") as f:
        f.write(mod_2_str)
    print("Saved to", safe_name_2 + ".mlir")
