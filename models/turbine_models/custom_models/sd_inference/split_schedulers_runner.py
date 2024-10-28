# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
import torch
from turbine_models.custom_models.sd_inference.schedulers_dyn import SchedulingModel, get_scheduler

torch.random.manual_seed(0)

def run_old_scheduler(
    device,
    sample,
    num_inference_steps,
    vmfb_path,
    unet_path,
    unet_params,
    unet_inps,
):
    runner = vmfbRunner(device, vmfb_path)
    unet = vmfbRunner(device, unet_path, unet_params)

    inputs0 = [
        ireert.asdevicearray(runner.config.device, sample, dtype="float16"),
    ]

    latents, add_time_ids, step_indexes, timesteps = runner.ctx.modules.compiled_scheduler["run_initialize"](*inputs0)
    guidance_scale = ireert.asdevicearray(runner.config.device, [7.5], dtype="float16")
    for i, t in enumerate(timesteps.to_host()):
        print("STEP: ")
        print(i)
        print("/")
        print(len(timesteps.to_host()))
        print("\n SCALED:")
        inputs1 = [
            latents,
            ireert.asdevicearray(runner.config.device, [i], dtype="int64"),
            timesteps
        ]
        latent_model_input, t_iter = runner.ctx.modules.compiled_scheduler["run_scale"](*inputs1)
        inputs2 = [
            latent_model_input,
            t_iter,
        ]
        inputs2.extend([ireert.asdevicearray(unet.config.device, x, dtype="float16") for x in unet_inps])
        inputs2.extend([add_time_ids, guidance_scale])
        print("\n DENOISE_INP:")
        print([i.to_host() for i in inputs2])
        if i == 1:
            denoise_inputs_s1 = [i.to_host() for i in inputs2]
        noise_pred = unet.ctx.modules.compiled_punet["main"](*inputs2)
        print("\n DENOISED:")
        print(noise_pred.to_host())
        inputs3 = [
            noise_pred,
            t_iter,
            latents
        ]
        latents = runner.ctx.modules.compiled_scheduler["run_step"](*inputs3)
        print("\n STEPPED:")
        print(latents.to_host())
    return denoise_inputs_s1


def run_dyn_scheduler(
    device,
    sample,
    num_inference_steps,
    vmfb_path,
    unet_path,
    unet_params,
    unet_inps
):
    runner = vmfbRunner(device, vmfb_path)
    unet = vmfbRunner(device, unet_path, unet_params)
    num_steps = ireert.asdevicearray(runner.config.device, [num_inference_steps], dtype="int64")


    inputs0 = [
        ireert.asdevicearray(runner.config.device, sample, dtype="float16"),
        num_steps
    ]

    latents, add_time_ids, timesteps, sigmas = runner.ctx.modules.compiled_scheduler["run_initialize"](*inputs0)
    guidance_scale = ireert.asdevicearray(runner.config.device, [7.5], dtype="float16")
    for i, t in enumerate(range(num_inference_steps)):
        print("STEP: ")
        print(i)
        print("/")
        print(num_inference_steps)
        print("\n SCALED:")
        step = ireert.asdevicearray(runner.config.device, [i], dtype="int64")
        inputs1 = [
            latents,
            step,
            timesteps,
            sigmas
        ]
        latent_model_input, t_iter, sigma, next_sigma = runner.ctx.modules.compiled_scheduler["run_scale"](*inputs1)
        inputs2 = [
            latent_model_input,
            t_iter,
        ]

        inputs2.extend([ireert.asdevicearray(unet.config.device, x, dtype="float16") for x in unet_inps])
        inputs2.extend([add_time_ids, guidance_scale])
        print("\n DENOISE_INP:")
        print([i.to_host() for i in inputs2])
        if i == 1:
            denoise_inputs_s1 = [i.to_host() for i in inputs2]
        noise_pred = unet.ctx.modules.compiled_punet["main"](*inputs2)
        print("\n DENOISED:")
        print(noise_pred.to_host())
        inputs3 = [
            noise_pred,
            latents,
            sigma,
            next_sigma
        ]
        latents = runner.ctx.modules.compiled_scheduler["run_step"](*inputs3)
        print("\n STEPPED:")
        print(latents.to_host())
    return denoise_inputs_s1



if __name__ == "__main__":
    from turbine_models.custom_models.sd_inference.sd_cmd_opts import args

    sample = torch.rand(
        args.batch_size, 4, 1024 // 8, 1024 // 8, dtype=torch.float16
    )
    device="hip"
    scheduler_id = "EulerDiscrete"
    dyn_vmfb_path = "stable_diffusion_xl_base_1_0_EulerDiscreteScheduler_bs1_1024x1024_fp16_gfx942.vmfb"
    st_vmfb_path = "stable_diffusion_xl_base_1_0_EulerDiscreteScheduler_bs1_1024x1024_fp16_3_gfx942.vmfb"
    unet_path = "stable_diffusion_xl_base_1_0_bs1_64_1024x1024_i8_punet_gfx942.vmfb"
    unet_params = "stable_diffusion_xl_base_1_0_punet_dataset_i8.irpa"
    num_inference_steps = 3
    prompt_embeds = torch.rand(2, 64, 2048)
    text_embeds = torch.rand(2, 1280)
    unet_inps = [prompt_embeds, text_embeds]
    s1_dyn = run_dyn_scheduler(
        device,
        sample,
        num_inference_steps,
        dyn_vmfb_path,
        unet_path,
        unet_params,
        unet_inps,
    )
    s1_old = run_old_scheduler(
        device,
        sample,
        num_inference_steps,
        st_vmfb_path,
        unet_path,
        unet_params,
        unet_inps,
    )
