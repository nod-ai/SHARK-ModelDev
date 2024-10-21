# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import inspect
from typing import List

import torch
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


class SharkSchedulerWrapper:
    def __init__(self, rt_device, vmfb):
        self.runner = vmfbRunner(rt_device, vmfb, None)

    def initialize(self, sample):
        sample, steps, timesteps = self.runner.ctx.modules.compiled_scheduler[
            "run_init"
        ](sample)
        return sample, steps.to_host(), timesteps

    def prep(self, sample, t, timesteps):
        return self.runner.ctx.modules.compiled_scheduler["run_prep"](
            sample, t, timesteps
        )

    def step(self, noise_pred, t, sample, guidance_scale, step_index):
        return self.runner.ctx.modules.compiled_scheduler["run_step"](
            noise_pred, t, sample, guidance_scale, step_index
        )


class FlowSchedulingModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        num_inference_steps,
        dtype,
    ):
        super().__init__()
        # For now, assumes SDXL implementation. May not need parametrization for other models,
        # but keeping hf_model_name in case.
        self.model = FlowMatchEulerDiscreteScheduler.from_pretrained(
            hf_model_name, subfolder="scheduler"
        )
        self.do_classifier_free_guidance = True
        self.model.set_timesteps(num_inference_steps)
        self.timesteps = self.model.timesteps
        self.dtype = dtype

    # TODO: Make steps dynamic here
    def initialize(self, sample):
        step_count = torch.tensor(len(self.timesteps))
        timesteps = self.model.timesteps
        # ops.trace_tensor("sample", sample[:,:,0,0])
        return (
            sample,
            step_count,
            timesteps.type(torch.float32),
        )

    def prepare_model_input(self, sample, t, timesteps):
        if self.do_classifier_free_guidance:
            latent_model_input = torch.cat([sample] * 2)
        else:
            latent_model_input = sample
        return latent_model_input.type(self.dtype), t.type(self.dtype)

    def step(self, noise_pred, t, sample, guidance_scale, i):
        self.model._step_index = i

        if self.do_classifier_free_guidance:
            noise_preds = noise_pred.chunk(2)
            noise_pred = noise_preds[0] + guidance_scale * (
                noise_preds[1] - noise_preds[0]
            )
        sample = self.model.step(noise_pred, t, sample, return_dict=False)[0]
        return sample.type(self.dtype)

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.model.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        eq = torch.eq(schedule_timesteps, timestep)
        eq = eq.int()
        index_candidates = torch.argmax(eq)
        index_candidates = index_candidates.unsqueeze(0)

        a = torch.numel(index_candidates)
        cond = torch.scalar_tensor(a)
        one = torch.scalar_tensor(1, dtype=torch.int64)
        zero = torch.scalar_tensor(0, dtype=torch.int64)
        index = torch.where(cond > 1, one, zero)
        index = index.unsqueeze(0)
        step_index = index_candidates.index_select(0, index)
        return step_index


# Wraps a diffusers scheduler running on native pytorch+cpu.
# This allows us to use it interchangeably with compiled schedulers in our pipeline(s).
class TorchCPUFlowSchedulerCompat:
    @torch.no_grad()
    def __init__(
        self, scheduler, batch_size, num_inference_steps, dest_device, latents_dtype
    ):
        self.do_classifier_free_guidance = True
        self.module = scheduler
        self.dest = dest_device
        self.dtype = latents_dtype
        self.batch_size = batch_size
        self.module.set_timesteps(num_inference_steps)
        self.timesteps = self.module.timesteps
        self.torch_dtype = (
            torch.float32 if latents_dtype == "float32" else torch.float16
        )

    def initialize(self, sample):
        if isinstance(sample, ireert.DeviceArray):
            sample = torch.tensor(sample.to_host(), dtype=torch.float32)
        step_indexes = torch.tensor(len(self.module.timesteps))
        return sample, step_indexes, self.module.timesteps

    def scale_model_input(self, sample, t, timesteps):
        if self.do_classifier_free_guidance:
            sample = torch.cat([sample] * 2)
        t = timesteps[t]
        t = t.expand(sample.shape[0])
        t = ireert.asdevicearray(self.dest, [t], self.dtype)
        sample = ireert.asdevicearray(self.dest, sample, self.dtype)
        return sample, t

    def step(self, noise_pred, t, latents, guidance_scale, i):
        if isinstance(t, ireert.DeviceArray):
            t = torch.tensor(t.to_host())
        if isinstance(guidance_scale, ireert.DeviceArray):
            guidance_scale = torch.tensor(guidance_scale.to_host())
        noise_pred = torch.tensor(noise_pred.to_host())
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        return self.module.step(
            noise_pred,
            t,
            latents,
            return_dict=False,
        )[0]


# Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete and adapted for dynamo compile.


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
# Only used for cpu scheduling.
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.no_grad()
def export_scheduler_model(
    hf_model_name: str,
    scheduler_id: str = "FlowEulerDiscrete",
    batch_size: int = 1,
    height: int = 512,
    width: int = 512,
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
    scheduler_module = FlowSchedulingModel(hf_model_name, num_inference_steps, dtype)
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

    do_classifier_free_guidance = True
    if do_classifier_free_guidance:
        init_batch_dim = 2
    else:
        init_batch_dim = 1

    sample = (
        batch_size,
        16,
        height // 8,
        width // 8,
    )
    noise_pred_shape = (
        batch_size * init_batch_dim,
        16,
        height // 8,
        width // 8,
    )
    example_init_args = [torch.empty(sample, dtype=dtype)]
    example_prep_args = (
        torch.empty(sample, dtype=dtype),
        torch.empty(1, dtype=torch.float32),
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
        torch.empty(1, dtype=dtype),
        torch.empty([1], dtype=torch.int64),
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
    def _prep(module, sample, t, timesteps):
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
            run_scale = _prep
            run_step = _step

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledScheduler(context=Context(), import_to=import_to)

    module = CompiledModule.get_mlir_module(inst)

    model_metadata_run_init = {
        "model_name": "sd3_scheduler_FlowEulerDiscrete",
        "input_shapes": [sample],
        "input_dtypes": [np_dtype],
        "output_shapes": [sample, "?", "?"],
        "output_dtypes": [np_dtype, "int32", "float32"],
    }
    model_metadata_run_prep = {
        "model_name": "sd3_scheduler_FlowEulerDiscrete",
        "input_shapes": [sample, (1,), ("?",)],
        "input_dtypes": [np_dtype, "float32", "float32"],
        "output_shapes": [noise_pred_shape, (1,)],
        "output_dtypes": [np_dtype, "float32"],
    }
    model_metadata_run_step = {
        "model_name": "sd3_scheduler_FlowEulerDiscrete",
        "input_shapes": [noise_pred_shape, (1,), sample, (1,), (1,)],
        "input_dtypes": [np_dtype, np_dtype, np_dtype, np_dtype, "int64"],
        "output_shapes": [sample],
        "output_dtypes": [np_dtype],
    }
    module = AddMetadataPass(module, model_metadata_run_init, "run_initialize").run()
    module = AddMetadataPass(module, model_metadata_run_prep, "run_scale").run()
    module = AddMetadataPass(module, model_metadata_run_step, "run_step").run()

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
