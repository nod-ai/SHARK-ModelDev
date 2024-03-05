import argparse
from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
import torch
import numpy as np
from tqdm.auto import tqdm

torch.random.manual_seed(0)


def run_unet(
    device,
    sample,
    timestep,
    prompt_embeds,
    text_embeds,
    time_ids,
    guidance_scale,
    vmfb_path,
    hf_model_name,
    hf_auth_token,
    external_weight_path,
    runner=None,
):
    if runner is None:
        runner = vmfbRunner(device, vmfb_path, external_weight_path)

    inputs = [
        ireert.asdevicearray(runner.config.device, sample),
        ireert.asdevicearray(runner.config.device, timestep),
        ireert.asdevicearray(runner.config.device, prompt_embeds),
        ireert.asdevicearray(runner.config.device, text_embeds),
        ireert.asdevicearray(runner.config.device, time_ids),
        ireert.asdevicearray(runner.config.device, guidance_scale),
    ]
    results = runner.ctx.modules.compiled_unet["main"](*inputs)

    return results


def run_unet_steps(
    device,
    sample,
    scheduler,
    prompt_embeds,
    text_embeds,
    time_ids,
    guidance_scale,
    vmfb_path,
    external_weight_path,
):
    runner = vmfbRunner(device, vmfb_path, external_weight_path)
    timestep = torch.zeros(1, dtype=torch.int64)
    inputs = [
        ireert.asdevicearray(runner.config.device, sample),
        ireert.asdevicearray(runner.config.device, timestep),
        ireert.asdevicearray(runner.config.device, prompt_embeds),
        ireert.asdevicearray(runner.config.device, text_embeds),
        ireert.asdevicearray(runner.config.device, time_ids),
        ireert.asdevicearray(runner.config.device, (guidance_scale,)),
    ]
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        timestep = t
        latent_model_input = scheduler.scale_model_input(sample, timestep)

        inputs[0] = latent_model_input = ireert.asdevicearray(
            runner.config.device, latent_model_input
        )
        inputs[1] = timestep = ireert.asdevicearray(
            runner.config.device, (timestep,), dtype="int64"
        )
        noise_pred = runner.ctx.modules.compiled_unet["main"](*inputs).to_host()
        sample = scheduler.step(
            torch.from_numpy(noise_pred).cpu(),
            timestep,
            sample,
            generator=None,
            return_dict=False,
        )[0]
    return sample


def run_torch_unet(
    hf_model_name,
    hf_auth_token,
    sample,
    timestep,
    prompt_embeds,
    text_embeds,
    time_ids,
    guidance_scale,
    precision="fp32",
):
    from turbine_models.custom_models.sdxl_inference.unet import UnetModel

    unet_model = UnetModel(
        hf_model_name,
        hf_auth_token,
        precision="fp32",
    )
    results = unet_model.forward(
        sample, timestep, prompt_embeds, text_embeds, time_ids, guidance_scale
    )
    np_torch_output = results.detach().cpu().numpy()
    return np_torch_output


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

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
    time_ids = torch.zeros(2 * args.batch_size, 6, dtype=dtype)
    guidance_scale = torch.tensor([7.5], dtype=dtype)

    turbine_output = run_unet(
        args.device,
        sample,
        timestep,
        prompt_embeds,
        text_embeds,
        time_ids,
        guidance_scale,
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

        torch_output = run_torch_unet(
            args.hf_model_name,
            args.hf_auth_token,
            sample.float(),
            timestep,
            prompt_embeds.float(),
            text_embeds.float(),
            time_ids.float(),
            guidance_scale.float(),
        )
        print("TORCH OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)
        err = utils.largest_error(torch_output, turbine_output)
        print("Largest Error: ", err)
        assert err < 9e-3

    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output = None
