# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import inspect
from typing import List

import torch
import math
from einops import repeat, rearrange
from typing import Any, Callable, Dict, List, Optional, Union
from shark_turbine.aot import *
import shark_turbine.ops.iree as ops
from shark_turbine.transforms.general.add_metadata import AddMetadataPass
from iree.compiler.ir import Context
import iree.runtime as ireert
import numpy as np

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
)

from turbine_models.turbine_tank import turbine_tank
from turbine_models.custom_models.sd_inference import utils
from turbine_models.model_runner import vmfbRunner


def time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, int(num_steps) + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


class FlowSchedulingModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        num_inference_steps,
        h,
        w,
        dtype,
    ):
        super().__init__()
        self.shift = hf_model_name != "flux-schnell"
        self.model = FlowMatchEulerDiscreteScheduler.from_pretrained(
            hf_model_name, subfolder="scheduler"
        )
        self.dtype = dtype
        self.timesteps = get_schedule(num_inference_steps, h * w // 4, shift=self.shift)


    def initialize(self, sample, steps):
        bs, c, h, w = sample.shape
        sample = rearrange(sample, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        timesteps = torch.tensor(self.timesteps, dtype=self.dtype)
        return (
            sample,
            steps,
            timesteps,
            img_ids,
        )


@torch.no_grad()
def export_scheduler_model(
    hf_model_name: str,
    scheduler_id: str = "FlowEulerDiscrete",
    batch_size: int = 1,
    height: int = 1024,
    width: int = 1024,
    shift: int = 1.0,
    num_inference_steps: int = 30,
    precision: str = "fp16",
    compile_to: str = "torch",
    device: str = None,
    target: str = None,
    ireec_flags: str = None,
    exit_on_vmfb: bool = False,
    pipeline_dir: str = None,
    input_mlir: str = None,
    upload_ir=False,
):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    np_dtype = "float16" if precision == "fp16" else "float32"
    sample = (
        batch_size,
        16,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
    )
    scheduler_module = FlowSchedulingModel(hf_model_name, num_inference_steps, sample[-2], sample[-1], dtype)
    vmfb_names = [
        "EulerFlowScheduler",
        f"bs{batch_size}_{height}x{width}",
        precision,
        str(num_inference_steps),
    ]
    vmfb_name = "_".join(vmfb_names)
    safe_name = utils.create_safe_name(hf_model_name, "_" + vmfb_name)
    if pipeline_dir:
        safe_name = os.path.join(pipeline_dir, safe_name)
    if input_mlir:
        vmfb_path = utils.compile_to_vmfb(
            input_mlir,
            device,
            target,
            ireec_flags,
            safe_name,
            mlir_source="file",
            return_path=not exit_on_vmfb,
        )
        return vmfb_path



    example_init_args = [
        torch.empty(sample, dtype=dtype),
        torch.empty(1, dtype=torch.int64),
    ]

    fxb = FxProgramsBuilder(scheduler_module)

    @fxb.export_program(
        args=(example_init_args,),
    )
    def _initialize(module, sample):
        return module.initialize(*sample)

    decomp_list = []
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):

        class CompiledScheduler(CompiledModule):
            run_initialize = _initialize

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledScheduler(context=Context(), import_to=import_to)

    module = CompiledModule.get_mlir_module(inst)

    module_str = str(module)
    if compile_to != "vmfb":
        return module_str
    elif compile_to == "vmfb":
        vmfb = utils.compile_to_vmfb(
            module_str,
            device,
            target,
            ireec_flags,
            safe_name,
            return_path=True,
        )
        if exit_on_vmfb:
            exit()
        return vmfb


if __name__ == "__main__":
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args

    mod_str = export_scheduler_model(
        args.hf_model_name,
        args.batch_size,
        args.height,
        args.width,
        args.shift,
        args.num_inference_steps,
        args.precision,
        args.compile_to,
        args.device,
        args.iree_target_triple,
        args.ireec_flags,
        exit_on_vmfb=False,
        input_mlir=args.input_mlir,
    )
    vmfb_names = [
        "EulerFlowScheduler",
        f"bs{args.batch_size}_{args.height}x{args.width}",
        args.precision,
        str(args.num_inference_steps),
        args.iree_target_triple,
    ]
    safe_name = "_".join(vmfb_names)
    if args.compile_to != "vmfb":
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
