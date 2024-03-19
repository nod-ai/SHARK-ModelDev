# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import re

from iree import runtime as ireert
import iree.compiler as ireec
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor
from turbine_models.turbine_tank import turbine_tank

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token, required"
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="CompVis/stable-diffusion-v1-4",
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
    upload_ir=False,
):
    if "google/t5" in hf_model_name:
        from transformers import T5Tokenizer, T5Model

        tokenizer = T5Tokenizer.from_pretrained(hf_model_name)
        text_encoder_model = T5Model.from_pretrained(hf_model_name)

    else:
        # TODO: Add better filtering mechanism for things that require CLIPProcessor
        if hf_model_name == "openai/clip-vit-large-patch14":
            tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            hf_subfolder = ""  # CLIPProcessor does not have a subfolder
        else:
            # Load the tokenizer and text encoder to tokenize and encode the text.
            tokenizer = CLIPTokenizer.from_pretrained(
                hf_model_name,
                subfolder="tokenizer",
                token=hf_auth_token,
            )
            hf_subfolder = "text_encoder"

        text_encoder_model = CLIPTextModel.from_pretrained(
            hf_model_name,
            subfolder=hf_subfolder,
            token=hf_auth_token,
        )

    mapper = {}
    utils.save_external_weights(
        mapper, text_encoder_model, external_weights, external_weight_path
    )

    if "google/t5" in hf_model_name:

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

            def main(
                self,
                inp=AbstractTensor(1, 77, dtype=torch.int64),
                decoder_input_ids=AbstractTensor(1, 77, dtype=torch.int64),
            ):
                return jittable(text_encoder_model.forward)(
                    input_ids=inp, decoder_input_ids=decoder_input_ids
                )

    else:

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

            def main(self, inp=AbstractTensor(1, 77, dtype=torch.int64)):
                return jittable(text_encoder_model.forward)(input_ids=inp)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledClip(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = utils.create_safe_name(hf_model_name, "-clip")
    if upload_ir:
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(module_str)
        model_name_upload = hf_model_name.replace("/", "_")
        model_name_upload += "-clip"
        turbine_tank.uploadToBlobStorage(
            str(os.path.abspath(f"{safe_name}.mlir")),
            f"{model_name_upload}/{model_name_upload}.mlir",
        )
    if compile_to != "vmfb":
        return module_str, tokenizer
    else:
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)


if __name__ == "__main__":
    args = parser.parse_args()
    mod_str, _ = export_clip_model(
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
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
