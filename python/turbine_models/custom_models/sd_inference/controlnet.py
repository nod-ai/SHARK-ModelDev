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
from diffusers import ControlNetModel as CNetModel

import safetensors
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token, required"
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="lllyasviel/control_v11p_sd15_canny",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for inference"
)
parser.add_argument(
    "--height", type=int, default=512, help="Height of Stable Diffusion"
)
parser.add_argument("--width", type=int, default=512, help="Width of Stable Diffusion")
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument("--external_weight_path", type=str, default="")
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="saves ir without global weights for size and readability, options [safetensors]",
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


class ControlNetModel(torch.nn.Module):
    def __init__(
        self, model_id="lllyasviel/control_v11p_sd15_canny", low_cpu_mem_usage=False
    ):
        super().__init__()
        self.cnet = CNetModel.from_pretrained(
            model_id,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        self.in_channels = self.cnet.config.in_channels
        self.train(False)

    def forward(
        self,
        latent,
        timestep,
        text_embedding,
        stencil_image_input,
    ):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        # TODO: guidance NOT NEEDED change in `get_input_info` later
        latents = torch.cat([latent] * 2)  # needs to be same as controlledUNET latents
        stencil_image = torch.cat(
            [stencil_image_input] * 2
        )  # needs to be same as controlledUNET latents
        (
            down_block_res_samples,
            mid_block_res_sample,
        ) = self.cnet.forward(
            latents,
            timestep,
            encoder_hidden_states=text_embedding,
            controlnet_cond=stencil_image,
            return_dict=False,
        )
        return tuple(list(down_block_res_samples) + [mid_block_res_sample])


def export_controlnet_model(
    controlnet_model,
    hf_model_name,
    batch_size,
    height,
    width,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target_triple=None,
    max_alloc=None,
):
    mapper = {}
    utils.save_external_weights(
        mapper, controlnet_model, external_weights, external_weight_path
    )

    class CompiledControlnet(CompiledModule):
        if external_weights:
            params = export_parameters(
                controlnet_model,
                external=True,
                external_scope="",
                name_mapper=mapper.get,
            )
        else:
            params = export_parameters(controlnet_model)

        def main(
            self,
            latent=AbstractTensor(1, 4, 512, 512, dtype=torch.float32),
            timestep=AbstractTensor(1, dtype=torch.float32),
            text_embedding=AbstractTensor(2, 72, 768, dtype=torch.float32),
            stencil_image_input=AbstractTensor(1, 3, 4096, 4096, dtype=torch.float32),
        ):
            return jittable(controlnet_model.forward)(
                latent,
                timestep,
                text_embedding,
                stencil_image_input,
            )

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledControlnet(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    if compile_to != "vmfb":
        return module_str
    else:
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)


if __name__ == "__main__":
    args = parser.parse_args()
    controlnet_model = ControlNetModel(
        args.hf_model_name,
    )
    mod_str = export_controlnet_model(
        controlnet_model,
        args.hf_model_name,
        args.batch_size,
        args.height,
        args.width,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.vulkan_max_allocation,
    )

    if mod_str is None:
        safe_name = args.hf_model_name.split("/")[-1].strip()
        safe_name = re.sub("-", "_", safe_name)
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
