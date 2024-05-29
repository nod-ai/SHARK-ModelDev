# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from typing import List

import torch
from shark_turbine.aot import *
from iree.compiler.ir import Context
import iree.runtime as ireert
import numpy as np

from diffusers import (
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDPMScheduler,
    DPMSolverSDEScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
)

from turbine_models.turbine_tank import turbine_tank
from turbine_models.custom_models.sd_inference import utils
from turbine_models.model_runner import vmfbRunner


class SharkSchedulerWrapper:
    def __init__(self, rt_device, vmfb, weights):
        self.runner = vmfbRunner(rt_device, vmfb, weights)

    def initialize(self, sample):
        return self.runner.ctx.modules.scheduler["initialize"](sample)

    def scale_model_input(self, sample, t):
        return self.runner.ctx.modules.scheduler["scale_model_input"](sample, t)

    def step(self, sample, latents, t):
        return self.runner.ctx.modules.scheduler["step"](sample, latents, t)


class SchedulingModel(torch.nn.Module):
    def __init__(self, scheduler, height, width, num_inference_steps, dtype):
        self.model = scheduler
        self.height = height
        self.width = width
        self.model.set_timesteps(num_inference_steps)
        self.model.is_scale_input_called = True
        self.dtype = dtype

    def initialize(self, sample):
        height = sample.shape[-2] * 8
        width = sample.shape[-1] * 8
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
        add_time_ids = add_time_ids.repeat(sample.shape[0], 1).type(self.dtype)
        timesteps = self.model.timesteps
        step_indexes = torch.tensor(len(timesteps))
        sample = sample * self.model.init_noise_sigma
        return sample.type(self.dtype), add_time_ids, step_indexes

    def scale_model_input(self, sample, t):
        return self.model.scale_model_input(sample, t)

    def step(self, latents, t, sample):
        return self.model.step(latents, t, sample)


class SharkSchedulerCPUWrapper:
    def __init__(self, pipe, scheduler):
        self.module = scheduler
        self.dest = pipe.runners["unet"].config.device
        self.dtype = pipe.iree_dtype

    def initialize(self, sample):
        sample, add_time_ids, step_indexes = self.module.initialize(
            torch.from_numpy(sample.to_host())
        )
        sample = ireert.asdevicearray(self.dest, sample, self.dtype)
        add_time_ids = ireert.asdevicearray(self.dest, add_time_ids, self.dtype)

        return sample, add_time_ids, step_indexes

    def scale_model_input(self, sample, t):
        scaled = ireert.asdevicearray(
            self.dest,
            self.module.scale_model_input(torch.from_numpy(sample.to_host()), t),
            self.dtype,
        )
        t = [self.module.model.timesteps[t]]
        t = ireert.asdevicearray(self.dest, t, self.dtype)
        return scaled, t

    def step(self, latents, t, sample):
        return ireert.asdevicearray(
            self.dest,
            self.module.step(
                torch.from_numpy(latents.to_host()),
                t,
                torch.from_numpy(sample.to_host()),
            ).prev_sample,
            self.dtype,
        )


def export_scheduler_model(
    hf_model_name: str,
    scheduler_id: str,
    batch_size: int = 1,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 30,
    precision: str = "fp16",
    compile_to: str = "torch",
    device: str = None,
    target_triple: str = None,
    ireec_flags: str = None,
    exit_on_vmfb: bool = False,
    pipeline_dir: str = None,
    input_mlir: str = None,
    upload_ir=False,
):
    scheduler = get_scheduler(hf_model_name, scheduler_id)
    scheduler_module = SchedulingModel(
        hf_model_name, scheduler, height, width, num_inference_steps
    )
    vmfb_name = (
        scheduler_id
        + "_"
        + f"{height}x{width}"
        + "_"
        + precision
        + "_"
        + str(num_inference_steps),
        +"_" + target_triple,
    )
    if pipeline_dir:
        safe_name = os.path.join(pipeline_dir, vmfb_name)
    else:
        safe_name = utils.create_safe_name(hf_model_name, vmfb_name)

    if input_mlir:
        vmfb_path = utils.compile_to_vmfb(
            input_mlir,
            device,
            target_triple,
            ireec_flags,
            safe_name,
            mlir_source="file",
            return_path=not exit_on_vmfb,
        )
        return vmfb_path

    dtype = torch.float16 if precision == "fp16" else torch.float32

    if precision == "fp16":
        scheduled_unet_model = scheduled_unet_model.half()

    sample = (
        batch_size,
        4,
        height // 8,
        width // 8,
    )

    class CompiledScheduler(CompiledModule):
        params = export_parameters(scheduled_unet_model)

        def initialize(
            self,
            sample=AbstractTensor(*sample, dtype=dtype),
        ):
            return jittable(scheduler_module.initialize)(sample)

        def scale_model_input(
            self,
            sample=AbstractTensor(*sample, dtype=dtype),
            t=AbstractTensor(1, dtype=dtype),
        ):
            return jittable(scheduler_module.scale_model_input)(sample, t)

        def step(
            self,
            sample=AbstractTensor(*sample, dtype=dtype),
            latents=AbstractTensor(1, dtype=dtype),
            t=AbstractTensor(1, dtype=dtype),
        ):
            return jittable(scheduler_module.step)(sample, latents, t)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledScheduler(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))

    if compile_to != "vmfb":
        return module_str
    elif compile_to == "vmfb":
        vmfb = utils.compile_to_vmfb(
            module_str,
            device,
            target_triple,
            ireec_flags,
            safe_name,
            return_path=True,
        )
        if exit_on_vmfb:
            exit()
        return vmfb


# from shark_turbine.turbine_models.schedulers import export_scheduler_model


def get_scheduler(model_id, scheduler_id):
    # TODO: switch over to turbine and run all on GPU
    print(f"\n[LOG] Initializing schedulers from model id: {model_id}")
    schedulers = {}
    for sched in SCHEDULER_MAP:
        schedulers[sched] = SCHEDULER_MAP[sched].from_pretrained(
            model_id, subfolder="scheduler"
        )
    schedulers["DPMSolverMultistep"] = DPMSolverMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler", algorithm_type="dpmsolver"
    )
    schedulers["DPMSolverMultistep++"] = DPMSolverMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler", algorithm_type="dpmsolver++"
    )
    schedulers[
        "DPMSolverMultistepKarras"
    ] = DPMSolverMultistepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["DPMSolverMultistepKarras"].config.use_karras_sigmas = True
    schedulers[
        "DPMSolverMultistepKarras++"
    ] = DPMSolverMultistepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        algorithm_type="dpmsolver++",
    )
    schedulers["DPMSolverMultistepKarras++"].config.use_karras_sigmas = True
    schedulers["DPMSolverSDE"] = DPMSolverSDEScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )
    return schedulers[scheduler_id]


SCHEDULER_MAP = {
    "PNDM": PNDMScheduler,
    "DDPM": DDPMScheduler,
    "KDPM2Discrete": KDPM2DiscreteScheduler,
    "LMSDiscrete": LMSDiscreteScheduler,
    "DDIM": DDIMScheduler,
    "LCMScheduler": LCMScheduler,
    "EulerDiscrete": EulerDiscreteScheduler,
    "EulerAncestralDiscrete": EulerAncestralDiscreteScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
    "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
}

if __name__ == "__main__":
    from turbine_models.custom_models.sd_inference.sd_cmd_opts import args

    mod_str = export_scheduler_model(
        args.hf_model_name,
        args.scheduler_id,
        args.batch_size,
        args.height,
        args.width,
        args.num_inference_steps,
        args.precision,
        args.compile_to,
        args.device,
        args.iree_target_triple,
        args.ireec_flags,
        exit_on_vmfb=False,
        input_mlir=args.input_mlir,
    )
    safe_name = utils.create_safe_name(
        args.hf_model_name,
        "_" + args.scheduler_id + "_" + str(args.num_inference_steps),
    )
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
