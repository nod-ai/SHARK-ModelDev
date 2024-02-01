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
parser.add_argument(
    "--controlled",
    dest="controlled",
    action="store_true",
    help="Whether or not to use controlled unet (for use with controlnet)",
)
parser.add_argument(
    "--no-controlled",
    dest="controlled",
    action="store_false",
    help="Whether or not to use controlled unet (for use with controlnet)",
)
parser.set_defaults(controlled=False)


class UnetModel(torch.nn.Module):
    def __init__(self, hf_model_name, hf_auth_token, is_controlled):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            hf_model_name,
            subfolder="unet",
            token=hf_auth_token,
        )
        self.guidance_scale = 7.5
        if is_controlled:
            self.forward = self.forward_controlled
        else:
            self.forward = self.forward_default

    def forward_default(self, sample, timestep, encoder_hidden_states):
        samples = torch.cat([sample] * 2)
        unet_out = self.unet.forward(
            samples, timestep, encoder_hidden_states, return_dict=False
        )[0]
        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred

    def forward_controlled(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        control1,
        control2,
        control3,
        control4,
        control5,
        control6,
        control7,
        control8,
        control9,
        control10,
        control11,
        control12,
        control13,
        scale1,
        scale2,
        scale3,
        scale4,
        scale5,
        scale6,
        scale7,
        scale8,
        scale9,
        scale10,
        scale11,
        scale12,
        scale13,
    ):
        db_res_samples = tuple(
            [
                control1 * scale1,
                control2 * scale2,
                control3 * scale3,
                control4 * scale4,
                control5 * scale5,
                control6 * scale6,
                control7 * scale7,
                control8 * scale8,
                control9 * scale9,
                control10 * scale10,
                control11 * scale11,
                control12 * scale12,
            ]
        )
        mb_res_samples = control13 * scale13
        samples = torch.cat([sample] * 2)
        unet_out = self.unet.forward(
            samples,
            timestep,
            encoder_hidden_states,
            down_block_additional_residuals=db_res_samples,
            mid_block_additional_residual=mb_res_samples,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred


def export_unet_model(
    unet_model,
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
    is_controlled=False,
):
    mapper = {}
    utils.save_external_weights(
        mapper, unet_model, external_weights, external_weight_path
    )

    encoder_hidden_states_sizes = (2, 77, 768)
    if hf_model_name == "stabilityai/stable-diffusion-2-1-base":
        encoder_hidden_states_sizes = (2, 77, 1024)

    sample = (batch_size, unet_model.unet.config.in_channels, height, width)

    class CompiledUnet(CompiledModule):
        if external_weights:
            params = export_parameters(
                unet_model, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(unet_model)

        def main(
            self,
            sample=AbstractTensor(*sample, dtype=torch.float32),
            timestep=AbstractTensor(1, dtype=torch.float32),
            encoder_hidden_states=AbstractTensor(
                *encoder_hidden_states_sizes, dtype=torch.float32
            ),
        ):
            return jittable(unet_model.forward)(sample, timestep, encoder_hidden_states)

    class CompiledControlledUnet(CompiledModule):
        if external_weights:
            params = export_parameters(
                unet_model, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(unet_model)

        def main(
            self,
            sample=AbstractTensor(*sample, dtype=torch.float32),
            timestep=AbstractTensor(1, dtype=torch.float32),
            encoder_hidden_states=AbstractTensor(
                *encoder_hidden_states_sizes, dtype=torch.float32
            ),
            control1=AbstractTensor(2, 320, height, width, dtype=torch.float32),
            control2=AbstractTensor(2, 320, height, width, dtype=torch.float32),
            control3=AbstractTensor(2, 320, height, width, dtype=torch.float32),
            control4=AbstractTensor(
                2, 320, height // 2, width // 2, dtype=torch.float32
            ),
            control5=AbstractTensor(
                2, 640, height // 2, width // 2, dtype=torch.float32
            ),
            control6=AbstractTensor(
                2, 640, height // 2, width // 2, dtype=torch.float32
            ),
            control7=AbstractTensor(
                2, 640, height // 4, width // 4, dtype=torch.float32
            ),
            control8=AbstractTensor(
                2, 1280, height // 4, width // 4, dtype=torch.float32
            ),
            control9=AbstractTensor(
                2, 1280, height // 4, width // 4, dtype=torch.float32
            ),
            control10=AbstractTensor(
                2, 1280, height // 8, width // 8, dtype=torch.float32
            ),
            control11=AbstractTensor(
                2, 1280, height // 8, width // 8, dtype=torch.float32
            ),
            control12=AbstractTensor(
                2, 1280, height // 8, width // 8, dtype=torch.float32
            ),
            control13=AbstractTensor(
                2, 1280, height // 8, width // 8, dtype=torch.float32
            ),
            scale1=AbstractTensor(1, dtype=torch.float32),
            scale2=AbstractTensor(1, dtype=torch.float32),
            scale3=AbstractTensor(1, dtype=torch.float32),
            scale4=AbstractTensor(1, dtype=torch.float32),
            scale5=AbstractTensor(1, dtype=torch.float32),
            scale6=AbstractTensor(1, dtype=torch.float32),
            scale7=AbstractTensor(1, dtype=torch.float32),
            scale8=AbstractTensor(1, dtype=torch.float32),
            scale9=AbstractTensor(1, dtype=torch.float32),
            scale10=AbstractTensor(1, dtype=torch.float32),
            scale11=AbstractTensor(1, dtype=torch.float32),
            scale12=AbstractTensor(1, dtype=torch.float32),
            scale13=AbstractTensor(1, dtype=torch.float32),
        ):
            return jittable(unet_model.forward)(
                sample,
                timestep,
                encoder_hidden_states,
                control1,
                control2,
                control3,
                control4,
                control5,
                control6,
                control7,
                control8,
                control9,
                control10,
                control11,
                control12,
                control13,
                scale1,
                scale2,
                scale3,
                scale4,
                scale5,
                scale6,
                scale7,
                scale8,
                scale9,
                scale10,
                scale11,
                scale12,
                scale13,
            )

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    if is_controlled:
        inst = CompiledControlledUnet(context=Context(), import_to=import_to)
    else:
        inst = CompiledUnet(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = utils.create_safe_name(hf_model_name, "-unet")
    if compile_to != "vmfb":
        return module_str
    else:
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)


if __name__ == "__main__":
    args = parser.parse_args()
    unet_model = UnetModel(
        args.hf_model_name if not args.controlled else "CompVis/stable-diffusion-v1-4",
        args.hf_auth_token,
        args.controlled,
    )
    mod_str = export_unet_model(
        unet_model,
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
        args.controlled,
    )
    safe_name = utils.create_safe_name(args.hf_model_name, "-unet")
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
