# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from typing import List

import torch
from shark_turbine.aot import *
import shark_turbine.ops.iree as ops
from shark_turbine.transforms.general.add_metadata import AddMetadataPass
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
    def __init__(self, rt_device, vmfb):
        self.runner = vmfbRunner(rt_device, vmfb, None)

    def initialize(self, sample):
        sample, time_ids, steps, timesteps = self.runner.ctx.modules.compiled_scheduler[
            "run_initialize"
        ](sample)
        return sample, time_ids, steps.to_host(), timesteps

    def scale_model_input(self, sample, t, timesteps):
        return self.runner.ctx.modules.compiled_scheduler["run_scale"](
            sample, t, timesteps
        )

    def step(self, noise_pred, t, sample, step_index):
        return self.runner.ctx.modules.compiled_scheduler["run_step"](
            noise_pred, t, sample, guidance_scale, step_index
        )


class SchedulingModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        scheduler,
        height,
        width,
        batch_size,
        num_inference_steps,
        dtype,
    ):
        super().__init__()
        # For now, assumes SDXL implementation. May not need parametrization for other models,
        # but keeping hf_model_name in case.
        self.model = scheduler
        self.height = height
        self.width = width
        self.is_sd3 = False
        if "stable-diffusion-3" in hf_model_name:
            self.is_sd3 = True
        self.batch_size = batch_size
        # Whether this will be used with CFG-enabled pipeline.
        self.do_classifier_free_guidance = True

        self.model.set_timesteps(num_inference_steps)
        self.timesteps = self.model.timesteps
        self.model.is_scale_input_called = True
        self.dtype = dtype

    # TODO: Make steps dynamic here
    def initialize(self, sample):
        height = self.height
        width = self.width
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype)
        if self.do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            add_time_ids = add_time_ids.repeat(self.batch_size, 1).type(self.dtype)
        step_count = torch.tensor(len(self.timesteps))
        timesteps = self.model.timesteps
        # ops.trace_tensor("timesteps", self.timesteps)
        sample = sample * self.model.init_noise_sigma
        return (
            sample.type(self.dtype),
            add_time_ids,
            step_count,
            timesteps.type(torch.float32),
        )

    def prepare_model_input(self, sample, i, timesteps):
        t = timesteps[i]

        latent_model_input = sample
        return self.model.scale_model_input(latent_model_input, t).type(
            self.dtype
        ), t.type(self.dtype)

    def step(self, noise_pred, t, sample):
        self.model._step_index = self.model.index_for_timestep(t)

        sample = self.model.step(noise_pred, t, sample, return_dict=False)[0]
        return sample.type(self.dtype)


class SharkSchedulerCPUWrapper:
    @torch.no_grad()
    def __init__(
        self,
        scheduler,
        batch_size,
        dest_device,
        latents_dtype,
        conditional_timesteps=False,
    ):
        self.module = scheduler
        self.dest = dest_device
        self.batch_size = batch_size
        self.timesteps = None
        self.do_classifier_free_guidance = True
        self.do_guidance = True
        self.repeat_sample = True

        # Enable this on init for models that use a pair of timestep values per unet step.
        # this includes sd3 and some others we don't support yet.
        # It allows passage of 'uncond_t' to the scale_model_input function and repeats the
        # default timestep value if no 'uncond_t' is passed.
        self.conditional_timesteps = conditional_timesteps

        self.dtype = latents_dtype
        self.torch_dtype = (
            torch.float32 if latents_dtype == "float32" else torch.float16
        )

    def initialize_sdxl(self, sample, num_inference_steps):
        if isinstance(sample, ireert.DeviceArray):
            sample = torch.tensor(sample.to_host(), dtype=torch.float32)

        self.module.set_timesteps(num_inference_steps)
        self.timesteps = self.module.timesteps
        height = sample.shape[2] * 8
        width = sample.shape[3] * 8
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.torch_dtype)
        if self.do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            add_time_ids = add_time_ids.repeat(self.batch_size, 1).type(
                self.torch_dtype
            )
        step_indexes = torch.tensor(len(self.timesteps))
        timesteps = self.timesteps
        sample = sample * self.module.init_noise_sigma
        return sample, add_time_ids, step_indexes, timesteps

    def initialize_sd(self, sample, num_inference_steps):
        if isinstance(sample, ireert.DeviceArray):
            sample = torch.tensor(sample.to_host(), dtype=torch.float32)
        self.module.set_timesteps(num_inference_steps)
        timesteps = self.module.timesteps
        sample = sample * self.module.init_noise_sigma
        return sample, timesteps

    def scale_model_input(self, sample, t, t_uncond=None):
        if self.repeat_sample:
            sample = torch.cat([sample] * 2)
        if self.conditional_timesteps:
            if t_uncond:
                t = torch.tensor([t, t_uncond])
            else:
                t = torch.tensor([t, t])
        else:
            t = torch.tensor([t])
        scaled = self.module.scale_model_input(sample, t)
        return scaled, t

    def step(self, noise_pred, t, latents, guidance_scale=None):
        if isinstance(t, ireert.DeviceArray):
            t = torch.tensor(t.to_host())
        if isinstance(noise_pred, ireert.DeviceArray):
            noise_pred = torch.tensor(noise_pred.to_host())
        elif isinstance(noise_pred, np.ndarray):
            noise_pred = torch.tensor(noise_pred)
        if isinstance(guidance_scale, ireert.DeviceArray):
            guidance_scale = torch.tensor(guidance_scale.to_host())
        if self.do_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        return self.module.step(
            noise_pred,
            t,
            latents,
        ).prev_sample


@torch.no_grad()
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
    target: str = None,
    ireec_flags: str = None,
    exit_on_vmfb: bool = False,
    pipeline_dir: str = None,
    input_mlir: str = None,
    attn_spec: str = None,
    external_weights: str = None,
    external_weight_path: str = None,
    upload_ir=False,
):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    iree_dtype = "float16" if precision == "fp16" else "float32"
    scheduler = get_scheduler(hf_model_name, scheduler_id)
    scheduler_module = SchedulingModel(
        hf_model_name, scheduler, height, width, batch_size, num_inference_steps, dtype
    )

    vmfb_names = [
        scheduler_id + "Scheduler",
        f"bs{batch_size}",
        f"{height}x{width}",
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

    sample = (
        batch_size,
        4,
        height // 8,
        width // 8,
    )
    noise_pred_shape = (
        batch_size,
        4,
        height // 8,
        width // 8,
    )
    example_init_args = [torch.empty(sample, dtype=dtype)]
    example_prep_args = (
        torch.empty(sample, dtype=dtype),
        torch.empty(1, dtype=torch.int64),
        torch.empty([19], dtype=torch.float32),
    )
    timesteps = torch.export.Dim("timesteps")
    prep_dynamic_args = {
        "sample": {},
        "t": {},
        "timesteps": {0: timesteps},
    }
    example_step_args = [
        torch.empty(noise_pred_shape, dtype=dtype),
        torch.empty(1, dtype=dtype),
        torch.empty(sample, dtype=dtype),
    ]

    fxb = FxProgramsBuilder(scheduler_module)

    @fxb.export_program(
        args=(example_init_args,),
    )
    def _initialize(module, sample):
        return module.initialize(*sample)

    @fxb.export_program(
        args=example_prep_args,
        dynamic_shapes=prep_dynamic_args,
    )
    def _scale(module, sample, t, timesteps):
        return module.prepare_model_input(sample, t, timesteps)

    @fxb.export_program(
        args=(example_step_args,),
    )
    def _step(module, inputs):
        return module.step(*inputs)

    decomp_list = []
    # if decomp_attn == True:
    #     decomp_list.extend(
    #         [
    #             torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
    #             torch.ops.aten._scaled_dot_product_flash_attention.default,
    #         ]
    #     )
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):

        class CompiledScheduler(CompiledModule):
            run_initialize = _initialize
            run_scale = _scale
            run_step = _step

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledScheduler(context=Context(), import_to=import_to)

    module = CompiledModule.get_mlir_module(inst)
    metadata_modelname = "_".join(
        [hf_model_name, scheduler_id, "scheduler", str(num_inference_steps)]
    )
    model_metadata_init = {
        "model_name": metadata_modelname,
        "input_shapes": [sample],
        "input_dtypes": [iree_dtype],
    }
    model_metadata_prep = {
        "model_name": metadata_modelname,
        "input_shapes": [sample, (1,), ("?",)],
        "input_dtypes": [iree_dtype, "int64", "float32"],
    }
    model_metadata_step = {
        "model_name": metadata_modelname,
        "input_shapes": [noise_pred_shape, (1,), sample],
        "input_dtypes": [iree_dtype, iree_dtype, iree_dtype],
    }
    module = AddMetadataPass(module, model_metadata_init, "run_initialize").run()
    module = AddMetadataPass(module, model_metadata_prep, "run_scale").run()
    module = AddMetadataPass(module, model_metadata_step, "run_step").run()
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
        return vmfb


def get_scheduler(model_id, scheduler_id):
    # TODO: switch over to turbine and run all on GPU
    print(f"\n[LOG] Initializing schedulers from model id: {model_id}")
    if scheduler_id in SCHEDULER_MAP.keys():
        scheduler = SCHEDULER_MAP[scheduler_id].from_pretrained(
            model_id, subfolder="scheduler"
        )
    elif all(x in scheduler_id for x in ["DPMSolverMultistep", "++"]):
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler", algorithm_type="dpmsolver++"
        )
    else:
        raise ValueError(f"Scheduler {scheduler_id} not found.")
    if "Karras" in scheduler_id:
        scheduler.config.use_karras_sigmas = True

    return scheduler


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
    "DPMSolverMultistepKarras": DPMSolverMultistepScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSDE": DPMSolverSDEScheduler,
    "DPMSolverSDEKarras": DPMSolverSDEScheduler,
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
    vmfb_names = [
        args.scheduler_id + "Scheduler",
        f"_bs{args.batch_size}_{args.height}x{args.width}",
        args.precision,
        str(args.num_inference_steps),
        args.iree_target_triple,
    ]
    safe_name = "_".join(vmfb_names)
    if args.compile_to != "vmfb":
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
