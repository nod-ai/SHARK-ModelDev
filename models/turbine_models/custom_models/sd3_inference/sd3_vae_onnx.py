# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
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
from diffusers import AutoencoderKL


class VaeModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            hf_model_name,
            subfolder="vae",
        )

    def forward(self, inp):
        inp = (inp / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(inp, return_dict=False)[0]
        image = image.float()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        return image

    # def decode(self, inp):
    #     inp = (inp / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    #     image = self.vae.decode(inp, return_dict=False)[0]
    #     image = image.float()
    #     image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    #     return image

    # def encode(self, inp):
    #     image_np = inp / 255.0
    #     image_np = np.moveaxis(image_np, 2, 0)
    #     batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
    #     image_torch = torch.from_numpy(batch_images)
    #     image_torch = 2.0 * image_torch - 1.0
    #     image_torch = image_torch
    #     latent = self.vae.encode(image_torch)
    #     return latent


def export_vae_model(
    vae_model,
    hf_model_name="stabilityai/stable-diffusion-3-medium-diffusers",
    batch_size=1,
    height=512,
    width=512,
    precision="fp32"
):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    file_prefix = "C:/Users/chiz/work/sd3/vae_decoder/exported/"
    safe_name = file_prefix + utils.create_safe_name(
        hf_model_name,
        f"_bs{batch_size}_{height}x{width}_{precision}_vae",
    ) + ".onnx"
    print(safe_name)

    if dtype == torch.float16:
        vae_model = vae_model.half()


    # input_image_shape = (height, width, 3)
    input_latents_shape = (batch_size, 16, height // 8, width // 8)
    input_latents = torch.empty(input_latents_shape, dtype=dtype)
    # encode_args = [
    #     torch.empty(
    #         input_image_shape,
    #         dtype=torch.float32,
    #     )
    # ]
    # decode_args = [
    #     torch.empty(
    #         input_latents_shape,
    #         dtype=dtype,
    #     )
    # ]

    torch.onnx.export(
                    vae_model,  # model being run
                    (
                        input_latents
                    ),  # model input (or a tuple for multiple inputs)
                    safe_name,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=17,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=[
                        "input_latents",
                    ],  # the model's input names
                    output_names=[
                        "sample_out",
                    ],  # the model's output names
                )
    return safe_name



if __name__ == "__main__":
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args

    vae_model = VaeModel(
            args.hf_model_name,
    )
    onnx_model_name = export_vae_model(
        vae_model,
        args.hf_model_name,
        1, # args.batch_size,
        512, # height=args.height,
        512, # width=args.width,
        "fp32" # precision=args.precision
    )
    print("Saved to", onnx_model_name)
