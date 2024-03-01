# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
import torch
from diffusers import (
    PNDMScheduler,
    UNet2DConditionModel,
)

parser = argparse.ArgumentParser()

# TODO move common runner flags to generic flag file
parser.add_argument(
    "--scheduler_id",
    type=str,
    help="Scheduler ID",
    default="PNDM",
)
parser.add_argument(
    "--num_inference_steps", type=int, default=50, help="Number of inference steps"
)
parser.add_argument(
    "--vmfb_path", type=str, default="", help="path to vmfb containing compiled module"
)
parser.add_argument(
    "--external_weight_path",
    type=str,
    default="",
    help="path to external weight parameters if model compiled without them",
)
parser.add_argument(
    "--compare_vs_torch",
    action="store_true",
    help="Runs both turbine vmfb and a torch model to compare results",
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="stabilityai/stable-diffusion-xl-base-1.0",
)
parser.add_argument(
    "--hf_auth_token",
    type=str,
    help="The Hugging face auth token, required for some models",
)
parser.add_argument(
    "--device",
    type=str,
    default="local-task",
    help="local-sync, local-task, cuda, vulkan, rocm",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for inference"
)
parser.add_argument(
    "--height", type=int, default=1024, help="Height of Stable Diffusion"
)
parser.add_argument("--width", type=int, default=1024, help="Width of Stable Diffusion")


def run_scheduler(
    device,
    sample,
    encoder_hidden_states,
    vmfb_path,
    hf_model_name,
    hf_auth_token,
    external_weight_path,
):
    runner = vmfbRunner(device, vmfb_path, external_weight_path)

    inputs = [
        ireert.asdevicearray(runner.config.device, sample),
        ireert.asdevicearray(runner.config.device, encoder_hidden_states),
    ]
    results = runner.ctx.modules.compiled_scheduler["main"](*inputs)
    return results


def run_sdxl_scheduler(
    device,
    sample,
    prompt_embeds,
    text_embeds,
    time_ids,
    vmfb_path,
    hf_model_name,
    hf_auth_token,
    external_weight_path,
):
    runner = vmfbRunner(device, vmfb_path, external_weight_path)

    inputs = [
        ireert.asdevicearray(runner.config.device, sample),
        ireert.asdevicearray(runner.config.device, prompt_embeds),
        ireert.asdevicearray(runner.config.device, text_embeds),
        ireert.asdevicearray(runner.config.device, time_ids),
    ]
    results = runner.ctx.modules.compiled_scheduler["main"](*inputs)
    return results


def run_torch_scheduler(
    hf_model_name,
    scheduler,
    num_inference_steps,
    sample,
    prompt_embeds,
    text_embeds,
    time_ids,
):
    class SDXLScheduler(torch.nn.Module):
        def __init__(
            self,
            hf_model_name,
            num_inference_steps,
            scheduler,
            hf_auth_token=None,
            precision="fp32",
        ):
            super().__init__()
            self.scheduler = scheduler
            self.scheduler.set_timesteps(num_inference_steps)
            self.guidance_scale = 7.5
            if precision == "fp16":
                try:
                    self.unet = UNet2DConditionModel.from_pretrained(
                        hf_model_name,
                        subfolder="unet",
                        auth_token=hf_auth_token,
                        low_cpu_mem_usage=False,
                        variant="fp16",
                    )
                except:
                    self.unet = UNet2DConditionModel.from_pretrained(
                        hf_model_name,
                        subfolder="unet",
                        auth_token=hf_auth_token,
                        low_cpu_mem_usage=False,
                    )
            else:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                )

        def forward(self, sample, prompt_embeds, text_embeds, time_ids):
            sample = sample * self.scheduler.init_noise_sigma
            for t in self.scheduler.timesteps:
                with torch.no_grad():
                    added_cond_kwargs = {
                        "text_embeds": text_embeds,
                        "time_ids": time_ids,
                    }
                    latent_model_input = torch.cat([sample] * 2)
                    t = t.unsqueeze(0)
                    # print('UNSQUEEZE T:', t)
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, timestep=t
                    )
                    noise_pred = self.unet.forward(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    sample = self.scheduler.step(
                        noise_pred, t, sample, return_dict=False
                    )[0]
            return sample

    scheduler_module = SDXLScheduler(
        hf_model_name,
        num_inference_steps,
        scheduler,
        hf_auth_token=None,
        precision="fp16",
    )
    results = scheduler_module.forward(sample, prompt_embeds, text_embeds, time_ids)
    np_torch_output = results.detach().cpu().numpy()
    return np_torch_output


if __name__ == "__main__":
    args = parser.parse_args()
    sample = torch.rand(
        args.batch_size, 4, args.height // 8, args.width // 8, dtype=torch.float32
    )
    if args.hf_model_name == "CompVis/stable-diffusion-v1-4":
        encoder_hidden_states = torch.rand(2, 77, 768, dtype=torch.float32)
    elif args.hf_model_name == "stabilityai/stable-diffusion-2-1-base":
        encoder_hidden_states = torch.rand(2, 77, 1024, dtype=torch.float32)

    sample = torch.rand(args.batch_size, 4, args.height // 8, args.width // 8)
    prompt_embeds = torch.rand(2, 77, 2048)
    text_embeds = torch.rand(2, 1280)
    time_ids = torch.rand(2, 6)
    turbine_output = run_sdxl_scheduler(
        args.device,
        sample,
        prompt_embeds,
        text_embeds,
        time_ids,
        args.vmfb_path,
        args.hf_model_name,
        args.hf_auth_token,
        args.external_weight_path,
    )
    print(
        "TURBINE OUTPUT:",
        turbine_output.to_host(),
        turbine_output.to_host().shape,
        turbine_output.to_host().dtype,
    )

    if args.compare_vs_torch:
        print("generating torch output: ")
        from turbine_models.custom_models.sd_inference import utils

        schedulers = utils.get_schedulers(args.hf_model_name)
        scheduler = schedulers[args.scheduler_id]
        torch_output = run_torch_scheduler(
            args.hf_model_name,
            scheduler,
            args.num_inference_steps,
            sample,
            prompt_embeds,
            text_embeds,
            time_ids,
        )
        print("TORCH OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)
        err = utils.largest_error(torch_output, turbine_output)
        print("Largest Error: ", err)
        assert err < 9e-3

    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output = None
