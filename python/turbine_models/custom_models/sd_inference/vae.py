# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree import runtime as ireert
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
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
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for inference"
)
parser.add_argument(
    "--height", type=int, default=512, help="Height of Stable Diffusion"
)
parser.add_argument("--width", type=int, default=512, help="Width of Stable Diffusion")
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


class VaeModel(torch.nn.Module):
    def __init__(self, hf_model_name, hf_auth_token):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            hf_model_name,
            subfolder="vae",
            token=hf_auth_token,
        )

    def forward(self, inp):
        with torch.no_grad():
            x = self.vae.decode(inp, return_dict=False)[0]
            return x


def export_vae_model(
    vae_model,
    hf_model_name,
    batch_size,
    height,
    width,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_file=None,
    device=None,
    target_triple=None,
    max_alloc=None,
):
    mapper = {}
    utils.save_external_weights(
        mapper, vae_model, external_weights, external_weight_file
    )

    sample = (batch_size, 4, height // 8, width // 8)

    class CompiledVae(CompiledModule):
        params = export_parameters(vae_model)

        def main(self, inp=AbstractTensor(*sample, dtype=torch.float32)):
            return jittable(vae_model.forward)(inp)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledVae(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = utils.create_safe_name(hf_model_name, "-vae")
    if compile_to != "vmfb":
        return module_str
    else:
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)


def run_vae_vmfb_comparison(vae_model, args):
    config = ireert.Config(args.device)

    if args.external_weight_file:
        index = ireert.ParameterIndex()
        index.load(args.external_weight_file)

    safe_name = utils.create_safe_name(args.hf_model_name, "-vae")
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
    inp = torch.rand(
        args.batch_size,
        4,
        args.height // 8,
        args.width // 8,
        dtype=torch.float32,
    )
    device_inputs = [ireert.asdevicearray(config.device, inp)]

    # Turbine output
    ModuleCompiled = ctx.modules.compiled_vae
    turbine_output = ModuleCompiled["main"](*device_inputs)
    print(
        "TURBINE OUTPUT:",
        turbine_output.to_host(),
        turbine_output.to_host().shape,
        turbine_output.to_host().dtype,
    )

    # Torch output
    torch_output = vae_model.forward(inp)
    torch_output = torch_output.detach().cpu().numpy()
    print("TORCH OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)

    err = utils.largest_error(torch_output, turbine_output)
    print("LARGEST ERROR:", err)
    assert err < 9e-5


if __name__ == "__main__":
    args = parser.parse_args()
    vae_model = VaeModel(
        args.hf_model_name,
        args.hf_auth_token,
    )
    if args.run_vmfb:
        run_vae_vmfb_comparison(vae_model, args)
    else:
        mod_str = export_vae_model(
            vae_model,
            args.hf_model_name,
            args.batch_size,
            args.height,
            args.width,
            args.hf_auth_token,
            args.compile_to,
            args.external_weights,
            args.external_weight_file,
            args.device,
            args.iree_target_triple,
            args.vulkan_max_allocation,
        )
        safe_name = utils.create_safe_name(args.hf_model_name, "-vae")
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
