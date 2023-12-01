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
from diffusers import UNet2DConditionModel

import safetensors
import argparse

parser = argparse.ArgumentParser()
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

class UnetModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(args.hf_model_name, subfolder="unet")
        self.guidance_scale = 7.5

    def forward(self, sample, timestep, encoder_hidden_states):
        samples = torch.cat([sample] * 2)
        unet_out = self.unet.forward(samples, timestep, encoder_hidden_states, return_dict=False)[0]
        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred


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


def export_unet_model(args, unet_model):
    mapper = {}
    save_external_weights(mapper, unet_model, args.external_weights, args.external_weight_file)

    if args.hf_model_name == "CompVis/stable-diffusion-v1-4":
        class CompiledUnet(CompiledModule):
            if args.external_weights:
                params = export_parameters(
                    unet_model, external=True, external_scope="", name_mapper=mapper.get
                )
            else:
                params = export_parameters(unet_model)

            def main(self, sample=AbstractTensor(1, 4, 64, 64, dtype=torch.float32),
                    timestep=AbstractTensor(1, dtype=torch.float32),
                    encoder_hidden_states=AbstractTensor(2, 77, 768, dtype=torch.float32)):
                return jittable(unet_model.forward)(
                    sample, timestep, encoder_hidden_states
                )
    elif args.hf_model_name == "stabilityai/stable-diffusion-2-1-base":
        class CompiledUnet(CompiledModule):
            if args.external_weights:
                params = export_parameters(
                    unet_model, external=True, external_scope="", name_mapper=mapper.get
                )
            else:
                params = export_parameters(unet_model)

            def main(self, sample=AbstractTensor(1, 4, 64, 64, dtype=torch.float32),
                    timestep=AbstractTensor(1, dtype=torch.float32),
                    encoder_hidden_states=AbstractTensor(2, 77, 1024, dtype=torch.float32)):
                return jittable(unet_model.forward)(
                    sample, timestep, encoder_hidden_states
                )
    

    import_to = "INPUT" if args.compile_to == "linalg" else "IMPORT"
    inst = CompiledUnet(context=Context(), import_to=import_to)

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


def run_unet_vmfb_comparison(args):
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
    sample = torch.rand(1, 4, 64, 64, dtype=torch.float32)
    timestep = torch.zeros(1, dtype=torch.float32)
    if args.hf_model_name == "CompVis/stable-diffusion-v1-4":
        encoder_hidden_states = torch.rand(2, 77, 768, dtype=torch.float32)
    elif args.hf_model_name == "stabilityai/stable-diffusion-2-1-base":
        encoder_hidden_states = torch.rand(2, 77, 1024, dtype=torch.float32)
    
    device_inputs = [ireert.asdevicearray(config.device, sample),
                    ireert.asdevicearray(config.device, timestep),
                    ireert.asdevicearray(config.device, encoder_hidden_states)]

    # Turbine output
    ModuleCompiled = ctx.modules.compiled_unet
    turbine_output = ModuleCompiled["main"](*device_inputs)
    print(turbine_output.to_host(), turbine_output.to_host().shape, turbine_output.to_host().dtype)

    # Torch output
    torch_output = unet_model.forward(sample, timestep, encoder_hidden_states)
    np_torch_output = torch_output.detach().cpu().numpy()
    print(np_torch_output, np_torch_output.shape, np_torch_output.dtype)

    err = largest_error(np_torch_output, turbine_output)
    print('LARGEST ERROR:', err)
    assert(err < 9e-5)


def largest_error(array1, array2):
    absolute_diff = np.abs(array1 - array2)
    max_error = np.max(absolute_diff)
    return max_error


if __name__ == "__main__":
    args = parser.parse_args()
    unet_model = UnetModel(args)
    if args.run_vmfb:
        run_unet_vmfb_comparison(args)
    else:
        mod_str = export_unet_model(args, unet_model)
        safe_name = args.hf_model_name.split("/")[-1].strip()
        safe_name = re.sub("-", "_", safe_name)
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")