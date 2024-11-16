# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import os
import sys
import math

import numpy as np
from shark_turbine.aot import *

from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from diffusers import SD3Transformer2DModel


class MMDiTModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name="stabilityai/stable-diffusion-3-medium-diffusers",
        dtype=torch.float16,
    ):
        super().__init__()
        self.mmdit = SD3Transformer2DModel.from_pretrained(
            hf_model_name,
            subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
    ):
        # timestep.expand(hidden_states.shape[0])
        noise_pred = self.mmdit(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            return_dict=False,
        )[0]
        return noise_pred


@torch.no_grad()
def export_mmdit_model(
    hf_model_name="stabilityai/stable-diffusion-3-medium-diffusers",
    batch_size=1,
    height=512,
    width=512,
    precision="fp16",
    max_length=77,
):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    mmdit_model = MMDiTModel(
        dtype=dtype,
    )
    file_prefix = "C:/Users/chiz/work/sd3/mmdit/exported/"
    safe_name = (
        file_prefix
        + utils.create_safe_name(
            hf_model_name,
            f"_bs{batch_size}_{max_length}_{height}x{width}_{precision}_mmdit",
        )
        + ".onnx"
    )
    print(safe_name)

    do_classifier_free_guidance = True
    init_batch_dim = 2 if do_classifier_free_guidance else 1
    batch_size = batch_size * init_batch_dim
    hidden_states_shape = (
        batch_size,
        16,
        height // 8,
        width // 8,
    )
    encoder_hidden_states_shape = (batch_size, 154, 4096)
    pooled_projections_shape = (batch_size, 2048)
    hidden_states = torch.empty(hidden_states_shape, dtype=dtype)
    encoder_hidden_states = torch.empty(encoder_hidden_states_shape, dtype=dtype)
    pooled_projections = torch.empty(pooled_projections_shape, dtype=dtype)
    timestep = torch.empty(batch_size, dtype=dtype)
    # mmdit_model(hidden_states, encoder_hidden_states, pooled_projections, timestep)

    torch.onnx.export(
        mmdit_model,  # model being run
        (
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
        ),  # model input (or a tuple for multiple inputs)
        safe_name,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=[
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
        ],  # the model's input names
        output_names=[
            "sample_out",
        ],  # the model's output names
    )
    return safe_name


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args

    onnx_model_name = export_mmdit_model(
        args.hf_model_name,
        1,  # args.batch_size,
        512,  # args.height,
        512,  # args.width,
        "fp16",  # args.precision,
        77,  # args.max_length,
    )

    print("Saved to", onnx_model_name)
