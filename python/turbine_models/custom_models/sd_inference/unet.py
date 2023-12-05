# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import re

from iree import runtime as ireert
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from diffusers import UNet2DConditionModel

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


class UnetModel(torch.nn.Module):
    def __init__(self, hf_model_name, hf_auth_token):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            hf_model_name,
            subfolder="unet",
            token=hf_auth_token,
        )
        self.guidance_scale = 7.5

    def forward(self, sample, timestep, encoder_hidden_states):
        samples = torch.cat([sample] * 2)
        unet_out = self.unet.forward(
            samples, timestep, encoder_hidden_states, return_dict=False
        )[0]
        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred


def export_unet_model(
    unet_model,
    hf_model_name,
    compile_to="torch",
    external_weights=None,
    external_weight_file=None,
):
    mapper = {}
    utils.save_external_weights(
        mapper, unet_model, external_weights, external_weight_file
    )

    encoder_hidden_states_sizes = (2, 77, 768)
    if hf_model_name == "stabilityai/stable-diffusion-2-1-base":
        encoder_hidden_states_sizes = (2, 77, 1024)

    class CompiledUnet(CompiledModule):
        if external_weights:
            params = export_parameters(
                unet_model, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(unet_model)

        def main(
            self,
            sample=AbstractTensor(1, 4, 64, 64, dtype=torch.float32),
            timestep=AbstractTensor(1, dtype=torch.float32),
            encoder_hidden_states=AbstractTensor(
                *encoder_hidden_states_sizes, dtype=torch.float32
            ),
        ):
            return jittable(unet_model.forward)(sample, timestep, encoder_hidden_states)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledUnet(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    if compile_to != "vmfb":
        return module_str
    else:
        utils.compile_to_vmfb(module_str, ["llvm-cpu"], safe_name)


def run_unet_vmfb_comparison(unet_model, args):
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

    device_inputs = [
        ireert.asdevicearray(config.device, sample),
        ireert.asdevicearray(config.device, timestep),
        ireert.asdevicearray(config.device, encoder_hidden_states),
    ]

    # Turbine output
    ModuleCompiled = ctx.modules.compiled_unet
    turbine_output = ModuleCompiled["main"](*device_inputs)
    print(
        "TURBINE OUTPUT:",
        turbine_output.to_host(),
        turbine_output.to_host().shape,
        turbine_output.to_host().dtype,
    )

    # Torch output
    torch_output = unet_model.forward(sample, timestep, encoder_hidden_states)
    np_torch_output = torch_output.detach().cpu().numpy()
    print(
        "TORCH OUTPUT:", np_torch_output, np_torch_output.shape, np_torch_output.dtype
    )

    err = utils.largest_error(np_torch_output, turbine_output)
    print("LARGEST ERROR:", err)
    assert err < 9e-5


if __name__ == "__main__":
    args = parser.parse_args()
    unet_model = UnetModel(
        args.hf_model_name,
        args.hf_auth_token,
    )
    if args.run_vmfb:
        run_unet_vmfb_comparison(unet_model, args)
    else:
        mod_str = export_unet_model(
            unet_model,
            args.hf_model_name,
            args.compile_to,
            args.external_weights,
            args.external_weight_file,
        )
        safe_name = args.hf_model_name.split("/")[-1].strip()
        safe_name = re.sub("-", "_", safe_name)
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
