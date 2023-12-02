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
import torch
import torch._dynamo as dynamo
from diffusers import AutoencoderKL

import safetensors
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


class VaeModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            args.hf_model_name,
            subfolder="vae",
            token=args.hf_auth_token,
        )

    def forward(self, inp):
        with torch.no_grad():
            x = self.vae.decode(inp, return_dict=False)[0]
            return x


def save_external_weights(
    mapper,
    model,
    external_weights=None,
    external_weight_file=None,
):
    if external_weights is not None:
        if external_weights == "safetensors":
            mod_params = dict(model.named_parameters())
            for name in mod_params:
                mapper["params." + name] = name
            if external_weight_file:
                safetensors.torch.save_file(mod_params, external_weight_file)
                print("Saved params to", external_weight_file)


def export_vae_model(args, vae_model):
    mapper = {}
    save_external_weights(
        mapper, vae_model, args.external_weights, args.external_weight_file
    )

    class CompiledVae(CompiledModule):
        params = export_parameters(vae_model)

        def main(self, inp=AbstractTensor(1, 4, 64, 64, dtype=torch.float32)):
            return jittable(vae_model.forward)(inp)

    import_to = "INPUT" if args.compile_to == "linalg" else "IMPORT"
    inst = CompiledVae(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    if args.compile_to != "vmfb":
        return module_str
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

        flatbuffer_blob = ireec.compile_str(
            module_str,
            target_backends=["llvm-cpu"],
            extra_args=flags,
        )
        with open(f"{safe_name}.vmfb", "wb+") as f:
            f.write(flatbuffer_blob)
        print("Saved to", safe_name + ".vmfb")
        exit()


def largest_error(array1, array2):
    absolute_diff = np.abs(array1 - array2)
    max_error = np.max(absolute_diff)
    return max_error


def run_vae_vmfb_comparison(args, vae_model):
    config = ireert.Config("local-task")

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
    inp = torch.rand(1, 4, 64, 64, dtype=torch.float32)
    device_inputs = [ireert.asdevicearray(config.device, inp)]

    # Turbine output
    ModuleCompiled = ctx.modules.compiled_vae
    turbine_output = ModuleCompiled["main"](*device_inputs)
    print(
        turbine_output.to_host(),
        turbine_output.to_host().shape,
        turbine_output.to_host().dtype,
    )

    # Torch output
    torch_output = vae_model.forward(inp)
    torch_output = torch_output.detach().cpu().numpy()
    print(torch_output, torch_output.shape, torch_output.dtype)

    err = largest_error(torch_output, turbine_output)
    print("LARGEST ERROR:", err)
    assert err < 9e-5


if __name__ == "__main__":
    args = parser.parse_args()
    vae_model = VaeModel(args)
    if args.run_vmfb:
        run_vae_vmfb_comparison(args, vae_model)
    else:
        mod_str = export_vae_model(args, vae_model)
        safe_name = args.hf_model_name.split("/")[-1].strip()
        safe_name = re.sub("-", "_", safe_name)
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")