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

from turbine_models.turbine_tank import turbine_tank
from turbine_models.custom_models.sd_inference import utils
from turbine_models.model_runner import vmfbRunner

class SharkSchedulerWrapper():
    def __init__(self, rt_device, vmfb, weights):
        self.runner = vmfbRunner(
            rt_device, vmfb, weights
        )
    
    def initialize(self, sample):
        return self.runner.ctx.modules.scheduler["initialize"](sample)
    
    def scale_model_input(self, sample, t):
        return self.runner.ctx.modules.scheduler["scale_model_input"](sample, t)
    
    def step(self, sample, latents, t):
        return self.runner.ctx.modules.scheduler["step"](sample, latents, t)


class SchedulingModel(torch.nn.Module):
    def __init__(self, scheduler, height, width):
        self.model = scheduler
        self.height = height
        self.width = width

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
        self.model.scale_model_input(sample, t)

    def step(self, sample, latents, t):
        self.model.step(self, sample, latents, t)

class SharkSchedulerCPUWrapper(SchedulingModel):
    def __init__(self, pipe, scheduler, height, width):
        super().__init__(scheduler, height, width)
        self.dest = pipe.runner["unet"].config.device
        self.dtype = pipe.iree_dtype
    
    def initialize(self, sample):
        for output in super().initialize(sample):
            iree_arrays = ireert.asdevicearray(self.dest, output, self.dtype)
        
        return iree_arrays
    
    def scale_model_input(self, sample, t):
        return ireert.asdevicearray(self.dest, super.scale_model_input(sample, t), self.dtype)
    
    def step(self, sample, latents, t):
        return ireert.asdevicearray(self.dest, super.step(sample.to_host(), latents.to_host(), t.to_host()), self.dtype)

def export_scheduler(
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
    schedulers = utils.get_schedulers(hf_model_name)
    scheduler = schedulers[scheduler_id]
    scheduler_module = SchedulingModel(
        hf_model_name, scheduler
    )
    vmfb_name = (
        scheduler_id
        + "_"
        + f"{height}x{width}"
        + "_"
        + precision
        + "_"
        + str(num_inference_steps),
        + "_"
        + target_triple
    )
    if pipeline_dir:
        safe_name = os.path.join(
            pipeline_dir, vmfb_name
        )
    else:
        safe_name = utils.create_safe_name(
            hf_model_name, vmfb_name
        )

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


if __name__ == "__main__":
    from turbine_models.custom_models.sd_inference.sd_cmd_opts import args

    mod_str = export_scheduler(
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
    safe_name = utils.create_safe_name(args.hf_model_name, "_" + args.scheduler_id + "_" + str(args.num_inference_steps))
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
