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
    default="stabilityai/sdxl-turbo",
)
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
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


def export_clip_model(
    hf_model_name,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target_triple=None,
    max_alloc=None,
):
    # Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        hf_model_name,
        subfolder="tokenizer",
        token=hf_auth_token,
    )
    text_encoder_1_model = CLIPTextModel.from_pretrained(
        hf_model_name,
        subfolder="text_encoder",
        token=hf_auth_token,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        hf_model_name,
        subfolder="tokenizer_2",
        token=hf_auth_token,
    )
    text_encoder_2_model = CLIPTextModelWithProjection.from_pretrained(
        hf_model_name,
        subfolder="text_encoder_2",
        token=hf_auth_token,
    )
    mapper = {}
    if external_weight_path:
        weights_path_1 = (
            external_weight_path.split(f".{external_weights}")[0]
            + "_1"
            + f".{external_weights}"
        )
        weights_path_2 = (
            external_weight_path.split(f".{external_weights}")[0]
            + "_2"
            + f".{external_weights}"
        )
    else:
        weights_path_1 = None
        weights_path_2 = None

    utils.save_external_weights(
        mapper, text_encoder_1_model, external_weights, weights_path_1
    )
    utils.save_external_weights(
        mapper, text_encoder_2_model, external_weights, weights_path_2
    )

    class CompiledClip1(CompiledModule):
        if external_weights:
            params = export_parameters(
                text_encoder_1_model,
                external=True,
                external_scope="",
                name_mapper=mapper.get,
            )
        else:
            params = export_parameters(text_encoder_1_model)

        def main(self, inp=AbstractTensor(1, 77, dtype=torch.int64)):
            return jittable(text_encoder_1_model.forward)(inp)

    class CompiledClip2(CompiledModule):
        if external_weights:
            params = export_parameters(
                text_encoder_2_model,
                external=True,
                external_scope="",
                name_mapper=mapper.get,
            )
        else:
            params = export_parameters(text_encoder_2_model)

        def main(self, inp=AbstractTensor(1, 77, dtype=torch.int64)):
            return jittable(text_encoder_2_model.forward)(inp)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst1 = CompiledClip1(context=Context(), import_to=import_to)
    inst2 = CompiledClip2(context=Context(), import_to=import_to)

    module_1_str = str(CompiledModule.get_mlir_module(inst1))
    module_2_str = str(CompiledModule.get_mlir_module(inst2))
    safe_name_1 = utils.create_safe_name(hf_model_name, "-clip-1")
    safe_name_2 = utils.create_safe_name(hf_model_name, "-clip-2")
    if compile_to != "vmfb":
        return module_1_str, module_2_str, tokenizer_1, tokenizer_2
    else:
        vmfb_path_1 = utils.compile_to_vmfb(
            module_1_str,
            device,
            target_triple,
            max_alloc,
            safe_name_1,
            return_path=True,
        )
        vmfb_path_2 = utils.compile_to_vmfb(
            module_2_str,
            device,
            target_triple,
            max_alloc,
            safe_name_2,
            return_path=True,
        )

        return vmfb_path_1, vmfb_path_2, tokenizer_1, tokenizer_2


if __name__ == "__main__":
    import re

    args = parser.parse_args()
    mod_1_str, mod_2_str, _, _ = export_clip_model(
        args.hf_model_name,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.vulkan_max_allocation,
    )
    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    safe_name_1 = safe_name + "_clip_1"
    safe_name_2 = safe_name + "_clip_2"
    with open(f"{safe_name_1}.mlir", "w+") as f:
        f.write(mod_1_str)
    print("Saved to", safe_name_1 + ".mlir")
    with open(f"{safe_name_2}.mlir", "w+") as f:
        f.write(mod_2_str)
    print("Saved to", safe_name_2 + ".mlir")
