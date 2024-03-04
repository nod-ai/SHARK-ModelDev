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
from shark_turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
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
    default="stabilityai/stable-diffusion-xl-base-1.0",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for inference"
)
parser.add_argument(
    "--height", type=int, default=1024, help="Height of Stable Diffusion"
)
parser.add_argument("--width", type=int, default=1024, help="Width of Stable Diffusion")
parser.add_argument(
    "--precision", type=str, default="fp16", help="Precision of Stable Diffusion"
)
parser.add_argument(
    "--max_length", type=int, default=77, help="Sequence Length of Stable Diffusion"
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
parser.add_argument(
    "--decomp_attn",
    default=False,
    action="store_true",
    help="Decompose attention at fx graph level",
)


class UnetModel(torch.nn.Module):
    def __init__(self, hf_model_name, hf_auth_token=None, precision="fp32"):
        super().__init__()
        if precision == "fp16":
            try:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                    variant="fp16",
                )
                print(f"GOT UNET FP16")
            except:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                )
                print(f"GOT UNET FP32")

        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                hf_model_name,
                subfolder="unet",
                auth_token=hf_auth_token,
                low_cpu_mem_usage=False,
            )
            print(f"GOT UNET FP32")

    def forward(
        self, sample, timestep, prompt_embeds, text_embeds, time_ids, guidance_scale
    ):
        with torch.no_grad():
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
            noise_pred = self.unet.forward(
                sample,
                timestep,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        return noise_pred


def export_unet_model(
    unet_model,
    hf_model_name,
    batch_size,
    height,
    width,
    precision="fp32",
    max_length=77,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target_triple=None,
    max_alloc=None,
    decomp_attn=False,
):
    mapper = {}
    decomp_list = DEFAULT_DECOMPOSITIONS
    if True or decomp_attn == True:
        decomp_list.extend(
            [
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
                torch.ops.aten._scaled_dot_product_flash_attention.default,
            ]
        )
    dtype = torch.float16 if precision == "fp16" else torch.float32
    if precision == "fp16":
        unet_model = unet_model.half()
    utils.save_external_weights(
        mapper, unet_model, external_weights, external_weight_path
    )
    sample = (
        2 * batch_size,
        unet_model.unet.config.in_channels,
        height // 8,
        width // 8,
    )
    time_ids_shape = (2 * batch_size, 6)
    prompt_embeds_shape = (2 * batch_size, max_length, 2048)
    text_embeds_shape = (2 * batch_size, 1280)

    class CompiledUnet(CompiledModule):
        if external_weights:
            params = export_parameters(
                unet_model, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(unet_model)

        def main(
            self,
            sample=AbstractTensor(*sample, dtype=dtype),
            timestep=AbstractTensor(1, dtype=torch.int64),
            prompt_embeds=AbstractTensor(*prompt_embeds_shape, dtype=dtype),
            text_embeds=AbstractTensor(*text_embeds_shape, dtype=dtype),
            time_ids=AbstractTensor(*time_ids_shape, dtype=dtype),
            guidance_scale=AbstractTensor(1, dtype=dtype),
        ):
            return jittable(unet_model.forward, decompose_ops=decomp_list)(
                sample, timestep, prompt_embeds, text_embeds, time_ids, guidance_scale
            )

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledUnet(context=Context(), import_to=import_to)

    # module_str = str(CompiledModule.get_mlir_module(inst))
    # safe_name = utils.create_safe_name(
    #     hf_model_name, f"_{max_length}_{height}x{width}_{precision}_unet_{device}"
    # )
    # with open(f"{safe_name}.mlir", "w+") as f:
    #     f.write(module_str)
    # print("Saved mlir to", safe_name + ".mlir")
    # if compile_to != "vmfb":
    #     return module_str
    # elif os.path.isfile(safe_name + ".vmfb"):
    #     exit()
    # else:
    #     utils.compile_to_vmfb(
    #         module_str,
    #         device,
    #         target_triple,
    #         max_alloc,
    #         safe_name,
    #         return_path=False,
    #     )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    unet_model = UnetModel(
        args.hf_model_name,
        args.hf_auth_token,
    )
    mod_str = export_unet_model(
        unet_model,
        args.hf_model_name,
        args.batch_size,
        args.height,
        args.width,
        args.precision,
        args.max_length,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.vulkan_max_allocation,
        args.decomp_attn,
    )
    # safe_name = utils.create_safe_name(
    #     args.hf_model_name,
    #     f"_{args.max_length}_{args.height}x{args.width}_{args.precision}_unet",
    # )
    # with open(f"{safe_name}.mlir", "w+") as f:
    #     f.write(mod_str)
    # print("Saved to", safe_name + ".mlir")
