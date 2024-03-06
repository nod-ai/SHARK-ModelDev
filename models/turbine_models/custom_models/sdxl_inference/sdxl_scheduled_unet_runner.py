import argparse
from turbine_models.model_runner import vmfbRunner
from turbine_models.custom_models.sd_inference import utils
from iree import runtime as ireert
import torch
import numpy as np
from tqdm.auto import tqdm

torch.random.manual_seed(0)


def run_unet_hybrid(
    sample,
    prompt_embeds,
    text_embeds,
    args,
):
    runner = vmfbRunner(args.rt_device, args.vmfb_path, args.external_weight_path)
    init_inp = [
        ireert.asdevicearray(runner.config.device, sample),
    ]
    sample, time_ids, steps = runner.ctx.modules.compiled_scheduled_unet[
        "run_initialize"
    ](
        *init_inp,
    )
    dtype = "float16" if args.precision == "fp16" else "float32"
    inputs = [
        sample,
        ireert.asdevicearray(runner.config.device, prompt_embeds),
        ireert.asdevicearray(runner.config.device, text_embeds),
        time_ids,
        ireert.asdevicearray(
            runner.config.device, np.asarray([args.guidance_scale]), dtype=dtype
        ),
        None,
    ]
    for i in range(0, steps.to_host()):
        inputs[0] = sample
        inputs[5] = ireert.asdevicearray(
            runner.config.device, torch.tensor([i]), dtype="int64"
        )
        sample = runner.ctx.modules.compiled_scheduled_unet["run_forward"](*inputs)
    return sample


def run_torch_scheduled_unet(
    sample,
    prompt_embeds,
    text_embeds,
    args,
):
    from diffusers import UNet2DConditionModel

    class ScheduledUnetModel(torch.nn.Module):
        def __init__(
            self,
            hf_model_name,
            scheduler_id,
            height,
            width,
            batch_size,
            hf_auth_token=None,
            precision="fp32",
            num_inference_steps=1,
        ):
            super().__init__()
            self.dtype = torch.float16 if precision == "fp16" else torch.float32
            self.scheduler = utils.get_schedulers(hf_model_name)[scheduler_id]
            original_size = (height, width)
            target_size = (height, width)
            crops_coords_top_left = (0, 0)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids, add_time_ids], dtype=self.dtype)
            self.add_time_ids = add_time_ids.repeat(batch_size * 1, 1)
            self.scheduler.set_timesteps(num_inference_steps)
            self._timesteps = self.scheduler.timesteps

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

        def initialize(self, sample):
            sample = sample * self.scheduler.init_noise_sigma
            return sample * self.scheduler.init_noise_sigma

        def forward(
            self, sample, prompt_embeds, text_embeds, guidance_scale, step_index
        ):
            with torch.no_grad():
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": self.add_time_ids,
                }
                t = self._timesteps[step_index]
                latent_model_input = torch.cat([sample] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
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
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                sample = self.scheduler.step(noise_pred, t, sample, return_dict=False)[
                    0
                ]
            return sample

    unet_model = ScheduledUnetModel(
        args.hf_model_name,
        args.scheduler_id,
        args.height,
        args.width,
        args.batch_size,
        args.hf_auth_token,
        args.precision,
        args.num_inference_steps,
    )
    sample = unet_model.initialize(sample)
    for i, t in tqdm(enumerate(unet_model.scheduler.timesteps)):
        timestep = t
        sample = unet_model.forward(
            sample.float(),
            prompt_embeds.float(),
            text_embeds.float(),
            args.guidance_scale,
            i,
        )
    return sample


def run_scheduled_unet(
    sample,
    prompt_embeds,
    text_embeds,
    args,
):
    pipe_runner = vmfbRunner(
        args.rt_device,
        [args.vmfb_path, args.pipeline_vmfb_path],
        [args.external_weight_path, None],
    )
    dtype = "float16" if args.precision == "fp16" else "float32"
    inputs = [
        ireert.asdevicearray(pipe_runner.config.device, sample),
        ireert.asdevicearray(pipe_runner.config.device, prompt_embeds),
        ireert.asdevicearray(pipe_runner.config.device, text_embeds),
        ireert.asdevicearray(
            pipe_runner.config.device, np.asarray([args.guidance_scale]), dtype=dtype
        ),
    ]
    print(inputs)
    latents = pipe_runner.ctx.modules.sdxl_compiled_pipeline["produce_image_latents"](
        *inputs,
    )

    return latents


def run_torch_diffusers_loop(
    sample,
    prompt_embeds,
    text_embeds,
    args,
):
    from turbine_models.custom_models.sdxl_inference.unet import UnetModel

    unet_model = UnetModel(
        args.hf_model_name,
        args.hf_auth_token,
        precision="fp32",
    )
    scheduler = utils.get_schedulers(args.hf_model_name)[args.scheduler_id]

    scheduler.set_timesteps(args.num_inference_steps)
    scheduler.is_scale_input_called = True
    sample = sample * scheduler.init_noise_sigma

    height = sample.shape[-2] * 8
    width = sample.shape[-1] * 8
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)

    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids, add_time_ids], dtype=torch.float32)
    add_time_ids = add_time_ids.repeat(args.batch_size * 1, 1)

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        timestep = t

        latent_model_input = scheduler.scale_model_input(sample, timestep)
        noise_pred = unet_model.forward(
            latent_model_input,
            timestep,
            prompt_embeds,
            text_embeds,
            add_time_ids,
            args.guidance_scale,
        )
        sample = scheduler.step(
            noise_pred,
            timestep,
            sample,
            return_dict=False,
        )[0]
    return sample.detach().cpu().numpy()


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args
    import numpy as np

    if args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    sample = torch.rand(
        args.batch_size, 4, args.height // 8, args.width // 8, dtype=dtype
    )
    timestep = torch.zeros(1, dtype=torch.int64)
    prompt_embeds = torch.rand(2 * args.batch_size, args.max_length, 2048, dtype=dtype)
    text_embeds = torch.rand(2 * args.batch_size, 1280, dtype=dtype)

    turbine_output = run_scheduled_unet(
        sample,
        prompt_embeds,
        text_embeds,
        args,
    )
    print(
        "TURBINE OUTPUT:",
        turbine_output.to_host(),
        turbine_output.to_host().shape,
        turbine_output.to_host().dtype,
    )

    if args.compare_vs_torch:
        from turbine_models.custom_models.sd_inference import utils

        print("generating output with python/torch scheduling unet: ")
        hybrid_output = run_unet_hybrid(
            sample,
            prompt_embeds,
            text_embeds,
            args,
        )
        print("generating torch output: ")
        torch_output = run_torch_scheduled_unet(
            sample,
            prompt_embeds,
            text_embeds,
            args,
        )
        print("generating torch+diffusers output: ")
        diff_output = run_torch_diffusers_loop(
            sample,
            prompt_embeds,
            text_embeds,
            args,
        )
        print(
            "diffusers-like OUTPUT:", diff_output, diff_output.shape, diff_output.dtype
        )
        print("torch OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)

        print(
            "HYBRID OUTPUT:",
            hybrid_output.to_host(),
            hybrid_output.to_host().shape,
            hybrid_output.to_host().dtype,
        )
        print("Comparing... \n(turbine pipelined unet to torch unet): ")
        try:
            np.testing.assert_allclose(
                turbine_output, torch_output, rtol=1e-2, atol=1e-4
            )
        except AssertionError as err:
            print(err)
        print("\n(turbine pipelined unet to hybrid unet): ")
        try:
            np.testing.assert_allclose(
                hybrid_output, turbine_output, rtol=1e-2, atol=1e-4
            )
            print("passed!")
        except AssertionError as err:
            print(err)
        print("\n(hybrid unet to diff unet): ")
        try:
            np.testing.assert_allclose(diff_output, hybrid_output, rtol=1e-2, atol=1e-4)
            print("passed!")
        except AssertionError as err:
            print(err)
        print("\n(turbine loop to diffusers loop): ")
        try:
            np.testing.assert_allclose(
                turbine_output, diff_output, rtol=1e-2, atol=1e-4
            )
            print("passed!")
        except AssertionError as err:
            print(err)
        print("\n(torch sched unet loop to diffusers loop): ")
        try:
            np.testing.assert_allclose(torch_output, diff_output, rtol=1e-2, atol=1e-4)
            print("passed!")
        except AssertionError as err:
            print(err)

    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output = None
