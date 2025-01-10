import argparse
from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
import torch
import numpy as np
from tqdm.auto import tqdm
from iree.runtime import BufferUsage

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
        ireert.asdevicearray(runner.config.device, sample, allowed_usage=BufferUsage.DEFAULT),
        ireert.asdevicearray(runner.config.device, timestep, allowed_usage=BufferUsage.DEFAULT),
        ireert.asdevicearray(runner.config.device, prompt_embeds, allowed_usage=BufferUsage.DEFAULT),
        ireert.asdevicearray(runner.config.device, text_embeds, allowed_usage=BufferUsage.DEFAULT),
        ireert.asdevicearray(runner.config.device, time_ids, allowed_usage=BufferUsage.DEFAULT),
        ireert.asdevicearray(runner.config.device, guidance_scale, allowed_usage=BufferUsage.DEFAULT)
    ]
    results = runner.ctx.modules.compiled_unet["run_forward"](*inputs)

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
        ireert.asdevicearray(runner.config.device, guidance_scale),
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
        noise_pred = runner.ctx.modules.compiled_unet["run_forward"](*inputs).to_host()
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

    save_inputs = True

    sample = torch.rand(
        args.batch_size, 4, args.height // 8, args.width // 8, dtype=dtype
    )
    timestep = torch.ones(1, dtype=dtype)
    prompt_embeds = torch.rand(2 * args.batch_size, args.max_length, 2048, dtype=dtype)
    text_embeds = torch.rand(2 * args.batch_size, 1280, dtype=dtype)
    time_ids = torch.rand(2 * args.batch_size, 6, dtype=dtype)
    guidance_scale = torch.tensor([7.5], dtype=dtype)

    if save_inputs:
        import os

        inputs_dir = "sdxl_unet_inputs_" + args.precision
        if not os.path.exists(inputs_dir):
            os.mkdir(inputs_dir)
        np.save("input1.npy", sample)
        np.save("input2.npy", timestep)
        np.save("input3.npy", prompt_embeds)
        np.save("input4.npy", text_embeds)
        np.save("input5.npy", time_ids)
        np.save("input6.npy", guidance_scale)

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
    ).to_host()
    print(
        "TURBINE OUTPUT:",
        turbine_output,
        turbine_output.shape,
        turbine_output.dtype,
    )

    if args.compare_vs_torch:
        print("generating torch output: ")
        from turbine_models.custom_models.sd_inference import utils

        # comment out .float for fp16... sorry.
        torch_output = run_torch_unet(
            args.hf_model_name,
            args.hf_auth_token,
            sample.float(),
            timestep,
            prompt_embeds.float(),
            text_embeds.float(),
            time_ids.float(),
            guidance_scale.float(),
            # precision="fp16",
        )
        print("TORCH OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)
        if save_inputs:
            np.save("golden_out.npy", torch_output)
        atol = 4e-2
        rtol = 4e-1
        np.testing.assert_allclose(turbine_output, torch_output, atol=atol, rtol=rtol)
