# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

from iree.compiler.ir import Context
import numpy as np
import re
from shark_turbine.aot import *
import torch
import torch._dynamo as dynamo
from torch._export import dynamic_dim
from torch._export.constraints import constrain_as_size, constrain_as_value
from transformers import CLIPTextModel, CLIPTokenizer

pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

import safetensors
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument("--external_weight_file", type=str, default="")
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="saves ir/vmfb without global weights for size and readability, options [safetensors]",
)

def export_transformer_model(
    compile_to,
    external_weights=None,
    external_weight_file=None,
):
    # Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer"
    )
    text_encoder_model = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder"
    )

    mapper = {}
    if external_weights is not None:
        if external_weights == "safetensors":
            mod_params = dict(text_encoder_model.named_parameters())
            for name in mod_params:
                mapper["params." + name] = name
            if external_weight_file:
                safetensors.torch.save_file(mod_params, external_weight_file)
                print("Saved params to", external_weight_file)


    class CompiledClip(CompiledModule):
        if external_weights:
            params = export_parameters(
                text_encoder_model, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(text_encoder_model)

        def main(self, inp=AbstractTensor(1, 77, dtype=torch.int64)):
            return jittable(text_encoder_model.forward)(
                inp
            )

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledClip(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    # TODO: for saving the vmfb
    safe_name = pretrained_model_name_or_path.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    if compile_to != "vmfb":
        return module_str, tokenizer
    else:
        flags = [
            "--iree-input-type=torch",
            "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            "--mlir-print-debuginfo",
            "--mlir-print-op-on-diagnostic=false",
            "--iree-llvmcpu-target-cpu-features=host",
            "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
            "--iree-llvmcpu-enable-microkernels",
            "--iree-llvmcpu-stack-allocation-limit=256000",
            "--iree-stream-resource-index-bits=64",
            "--iree-vm-target-index-bits=64",
            "--iree-vm-bytecode-module-strip-source-map=true",
            "--iree-util-zero-fill-elided-attrs",
            "--iree-vm-target-truncate-unsupported-floats",
            "--iree-codegen-check-ir-before-llvm-conversion=false",
            "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            "--iree-opt-const-expr-hoisting=False",
        ]

        import iree.compiler as ireec

        flatbuffer_blob = ireec.compile_str(
            module_str,
            target_backends=["llvm-cpu"],
            extra_args=flags,
        )
        with open(f"{safe_name}.vmfb", "wb+") as f:
            f.write(flatbuffer_blob)
        print("Saved to", safe_name + ".vmfb")
        exit()


if __name__ == "__main__":
    args = parser.parse_args()
    mod_str, _ = export_transformer_model(
        args.compile_to,
        args.external_weights,
        args.external_weight_file,
    )
    safe_name = pretrained_model_name_or_path.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")