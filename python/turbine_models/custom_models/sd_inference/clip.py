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
from transformers import CLIPTextModel, CLIPTokenizer

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
parser.add_argument("--run_vmfb", action="store_true")
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument("--external_weight_file", type=str, default="")
parser.add_argument("--vmfb_path", type=str, default="")
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

prompt = ["a photograph of an astronaut riding a horse"]


def export_clip_model(
    hf_model_name,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_file=None,
    device=None,
    target_triple=None,
    max_alloc=None,
):
    # Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(
        hf_model_name,
        subfolder="tokenizer",
        token=hf_auth_token,
    )
    text_encoder_model = CLIPTextModel.from_pretrained(
        hf_model_name,
        subfolder="text_encoder",
        token=hf_auth_token,
    )

    mapper = {}
    utils.save_external_weights(
        mapper, text_encoder_model, external_weights, external_weight_file
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

        def main(self, inp=AbstractTensor(1, 77, dtype=torch.int64)):
            return jittable(text_encoder_model.forward)(inp)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledClip(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    if compile_to != "vmfb":
        return module_str, tokenizer
    else:
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)


def run_clip_vmfb_comparison(args):
    config = ireert.Config(args.device)

    if args.external_weight_file:
        index = ireert.ParameterIndex()
        index.load(args.external_weight_file)

    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    if args.vmfb_path:
        mod = ireert.VmModule.mmap(config.vm_instance, args.vmfb_path)
    elif os.path.exists(f"{safe_name}.vmfb"):
        mod = ireert.VmModule.mmap(config.vm_instance, f"{safe_name}.vmfb")
    else:
        sys.exit("no vmfb_path provided, required for run_vmfb")

    vm_modules = [
        mod,
        ireert.create_hal_module(config.vm_instance, config.device),
    ]
    if args.external_weight_file:
        param_module = ireert.create_io_parameters_module(
            config.vm_instance, index.create_provider(scope="model")
        )
        vm_modules.insert(0, param_module)

    ctx = ireert.SystemContext(
        vm_modules=vm_modules,
        config=config,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.hf_model_name,
        subfolder="tokenizer",
        token=args.hf_auth_token,
    )
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    inp = text_input.input_ids
    device_inputs = [ireert.asdevicearray(config.device, inp)]

    # Turbine output
    ModuleCompiled = ctx.modules.compiled_clip
    turbine_outputs = ModuleCompiled["main"](*device_inputs)
    turbine_output = turbine_outputs[0]
    print(
        "TURBINE OUTPUT:",
        turbine_output.to_host(),
        turbine_output.to_host().shape,
        turbine_output.to_host().dtype,
    )

    # Torch output
    text_encoder_model = CLIPTextModel.from_pretrained(
        args.hf_model_name,
        subfolder="text_encoder",
        token=args.hf_auth_token,
    )
    torch_output = text_encoder_model.forward(inp)[0]
    np_torch_output = torch_output.detach().cpu().numpy()
    print(
        "TORCH OUTPUT:", np_torch_output, np_torch_output.shape, np_torch_output.dtype
    )

    err = utils.largest_error(np_torch_output, turbine_output)
    print("LARGEST ERROR:", err)
    assert err < 9e-5


if __name__ == "__main__":
    args = parser.parse_args()
    if args.run_vmfb:
        run_clip_vmfb_comparison(args)
    else:
        mod_str, _ = export_clip_model(
            args.hf_model_name,
            args.hf_auth_token,
            args.compile_to,
            args.external_weights,
            args.external_weight_file,
            args.device,
            args.iree_target_triple,
            args.vulkan_max_allocation,
        )
        safe_name = args.hf_model_name.split("/")[-1].strip()
        safe_name = re.sub("-", "_", safe_name)
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
