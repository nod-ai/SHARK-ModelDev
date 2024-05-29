# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re

from iree.compiler.ir import Context
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
import torch
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
    hf_auth_token: str = None,
    max_length: int = 64,
    precision: str = "fp16",
    compile_to: str = "torch",
    external_weights: str = None,
    external_weight_path: str = None,
    device: str = "llvm-cpu",
    target_triple: str = "x86_64-linux-gnu",
    ireec_flags: str = None,
    exit_on_vmfb: bool = False,
    pipeline_dir: str = None,
    input_mlir: str = None,
    td_spec: str = None,
    weights_only: bool = False,
    upload_ir: bool = False,
):
    input_len = max_length
    if pipeline_dir not in [None, ""]:
        safe_name = os.path.join(pipeline_dir, "clip")
    else:
        safe_name = utils.create_safe_name(
            hf_model_name, f"_{str(max_length)}-{precision}-clip-{device}"
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
            const_expr_hoisting=True,
            attn_spec=td_spec,
        )
        return vmfb_path
    if "google/t5" in hf_model_name:
        from transformers import T5Tokenizer, T5Model

        tokenizer = T5Tokenizer.from_pretrained(hf_model_name)
        text_encoder_model = T5Model.from_pretrained(hf_model_name)
        input_len = 512

    else:
        # TODO: Add better filtering mechanism for things that require CLIPProcessor
        if "openai" in hf_model_name:
            tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            hf_subfolder = ""  # CLIPProcessor does not have a subfolder
            input_len = 10
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

    if weights_only:
        return external_weight_path

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
                inp=AbstractTensor(1, input_len, dtype=torch.int64),
                decoder_input_ids=AbstractTensor(1, input_len, dtype=torch.int64),
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

            def main(self, inp=AbstractTensor(1, input_len, dtype=torch.int64)):
                return jittable(text_encoder_model.forward)(input_ids=inp)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledClip(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    if compile_to != "vmfb":
        return module_str, tokenizer
    else:
        vmfb_path = utils.compile_to_vmfb(
            module_str,
            device,
            target_triple,
            ireec_flags,
            safe_name,
            return_path=not exit_on_vmfb,
            const_expr_hoisting=True,
            attn_spec=td_spec,
        )
        return vmfb_path, None


if __name__ == "__main__":
    from .sd_cmd_opts import args

    mod_str, _ = export_clip_model(
        args.hf_model_name,
        args.max_length,
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
        td_spec=args.attn_spec,
        weights_only=False,
        upload_ir=False,
    )
    if args.input_mlir:
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name, f"{str(args.max_length)}_{args.precision}_clip"
    )
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
