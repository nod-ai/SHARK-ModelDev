import argparse
from turbine_models.model_runner import vmfbRunner
from turbine_models.custom_models.sd_inference import utils
from iree import runtime as ireert
import torch
import numpy as np
from tqdm.auto import tqdm

torch.random.manual_seed(0)

def run_torch_scheduled_unet(
    sample,
    prompt_embeds,
    text_embeds,
    args,
):
    from turbine_models.custom_models.sdxl_inference.sdxl_scheduled_unet import (
        SDXLScheduledUnet,
    )

    unet_model = SDXLScheduledUnet(
        args.hf_model_name,
        args.scheduler_id,
        args.height,
        args.width,
        args.batch_size,
        args.hf_auth_token,
        "fp32",
        args.num_inference_steps,
    ).float()
    sample, add_time_ids, steps = unet_model.initialize(sample)
    for i in range(steps):
        sample = unet_model.forward(
            sample.float(),
            prompt_embeds.float(),
            text_embeds.float(),
            add_time_ids.float(),
            args.guidance_scale,
            i,
        )
    return sample


def run_scheduled_unet_compiled(
    sample,
    prompt_embeds,
    text_embeds,
    args,
):
    pipe_runner = vmfbRunner(
        args.device,
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

def run_scheduled_unet_python(
    sample,
    prompt_embeds,
    text_embeds,
    args,
):
    unet_runner = vmfbRunner(
        args.device,
        args.vmfb_path,
        args.external_weight_path,
    )
    dtype = "float16" if args.precision == "fp16" else "float32"
    sample, time_ids, steps = run_scheduled_unet_initialize(
        sample,
        unet_runner,
        args,
    )
    iree_inputs = [
        sample,
        ireert.asdevicearray(unet_runner.config.device, prompt_embeds),
        ireert.asdevicearray(unet_runner.config.device, text_embeds),
        time_ids,
        ireert.asdevicearray(
            unet_runner.config.device, np.asarray([args.guidance_scale]), dtype=dtype
        ),
        None,
    ]
    for i in range(steps.to_host()):
        iree_inputs[0] = sample
        iree_inputs[5] = ireert.asdevicearray(
            unet_runner.config.device, torch.tensor([i]), dtype="int64"
        )
        sample = run_scheduled_unet_forward(
            sample,
            prompt_embeds,
            text_embeds,
            time_ids,
            args.guidance_scale,
            i,
            unet_runner,
            args,
        )
    return sample

def run_scheduled_unet_initialize(
    sample,
    unet_runner,
    args,
):
    dtype = "float16" if args.precision == "fp16" else "float32"
    inputs = [
        ireert.asdevicearray(unet_runner.config.device, sample),
    ]
    sample, time_ids, steps = unet_runner.ctx.modules.compiled_scheduled_unet["run_initialize"](
        *inputs,
    )
    return sample, time_ids, steps

def run_scheduled_unet_forward(
    sample,
    prompt_embeds,
    text_embeds,
    time_ids,
    guidance_scale,
    timestep,
    unet_runner,
    args,
):
    dtype = "float16" if args.precision == "fp16" else "float32"
    inputs = [
        ireert.asdevicearray(unet_runner.config.device, sample, dtype=dtype),
        ireert.asdevicearray(unet_runner.config.device, prompt_embeds, dtype=dtype),
        ireert.asdevicearray(unet_runner.config.device, text_embeds, dtype=dtype),
        time_ids,
        ireert.asdevicearray(
            unet_runner.config.device, np.asarray([guidance_scale]), dtype=dtype
        ),
        ireert.asdevicearray(
            unet_runner.config.device, np.asarray([timestep]), dtype="int64"
        ),
    ]
    sample = unet_runner.ctx.modules.compiled_scheduled_unet["run_forward"](*inputs)
    return sample


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
    sample = sample.to(torch.float32)
    prompt_embeds = prompt_embeds.to(torch.float32)
    text_embeds = text_embeds.to(torch.float32)

    for idx, i in enumerate(scheduler.timesteps):
        timestep = i

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

    init_batch_dim = 2
    sample = torch.rand(
        args.batch_size, 4, args.height // 8, args.width // 8, dtype=dtype
    )
    timestep = torch.zeros(1, dtype=torch.int64)
    prompt_embeds = torch.rand(
        init_batch_dim * args.batch_size, args.max_length, 2048, dtype=dtype
    )
    text_embeds = torch.rand(init_batch_dim * args.batch_size, 1280, dtype=dtype)
    time_ids = torch.rand(init_batch_dim * args.batch_size, 6)
    if args.compiled_pipeline:
        assert args.pipeline_vmfb_path is not None, "--pipeline_vmfb_path is required for compiled pipeline run"
        turbine_compiled_output = run_scheduled_unet_compiled(
            sample,
            prompt_embeds,
            text_embeds,
            args,
        ).to_host()
        print(
            "TURBINE COMPILED OUTPUT:",
            turbine_compiled_output,
            turbine_compiled_output.shape,
            turbine_compiled_output.dtype,
        )

    turbine_python_output = run_scheduled_unet_python(
        sample,
        prompt_embeds,
        text_embeds,
        args,
    ).to_host()
    print(
        "TURBINE PYTHON OUTPUT:",
        turbine_python_output,
        turbine_python_output.shape,
        turbine_python_output.dtype,
    )



    if args.compare_vs_torch:
        from turbine_models.custom_models.sd_inference import utils

        print("generating torch output: ")
        torch_output = run_torch_scheduled_unet(
            sample,
            prompt_embeds,
            text_embeds,
            args,
        )
        print("torch OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)

        print("\n(torch sched unet loop to iree python loop): ")
        try:
            np.testing.assert_allclose(turbine_python_output, torch_output, rtol=4e-2, atol=4e-2)
            print("passed!")
        except AssertionError as err:
            print(err)

        if args.compiled_pipeline:
            print("\n(torch sched unet loop to iree compiled loop): ")
            try:
                np.testing.assert_allclose(turbine_compiled_output, torch_output, rtol=4e-2, atol=4e-2)
                print("passed!")
            except AssertionError as err:
                print(err)
