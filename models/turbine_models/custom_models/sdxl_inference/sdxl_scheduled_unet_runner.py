import argparse
from turbine_models.model_runner import vmfbRunner
from turbine_models.custom_models.sd_inference import utils, schedulers
from iree import runtime as ireert
import torch
import numpy as np
from tqdm.auto import tqdm
from iree.turbine.ops.iree import trace_tensor

torch.random.manual_seed(0)


@torch.no_grad()
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
            torch.tensor(args.guidance_scale, dtype=torch.float32),
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
    latents = pipe_runner.ctx.modules.sdxl_compiled_pipeline["produce_image_latents"](
        *inputs,
    )

    return latents


def run_scheduled_unet_initialize(
    sample,
    unet_runner,
    args,
):
    inputs = [
        ireert.asdevicearray(unet_runner.config.device, sample),
    ]
    sample, time_ids, steps = unet_runner.ctx.modules.compiled_scheduled_unet[
        "run_initialize"
    ](
        *inputs,
    )
    return sample, time_ids, steps


def run_scheduled_unet_forward(
    inputs,
    unet_runner,
    args,
):
    sample = unet_runner.ctx.modules.compiled_scheduled_unet["run_forward"](*inputs)
    return sample


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
            iree_inputs,
            unet_runner,
            args,
        )
    return sample


def run_unet_split_scheduled(
    sample,
    prompt_embeds,
    text_embeds,
    args,
):
    dtype = "float16" if args.precision == "fp16" else "float32"
    torch_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    unet_runner = vmfbRunner(
        args.device,
        args.vmfb_path,
        args.external_weight_path,
    )
    if not args.scheduler_vmfb_path:
        print("--scheduler_vmfb_path not supplied. Using cpu scheduling.")
        scheduler = schedulers.get_scheduler(args.hf_model_name, args.scheduler_id)
        scheduler = schedulers.SharkSchedulerCPUWrapper(
            scheduler,
            args.batch_size,
            args.num_inference_steps,
            unet_runner.config.device,
            dtype,
        )
        guidance_scale = torch.tensor([args.guidance_scale])
    else:
        scheduler = schedulers.SharkSchedulerWrapper(
            args.device,
            args.scheduler_vmfb_path,
        )
        guidance_scale = ireert.asdevicearray(
            scheduler.runner.config.device,
            np.asarray([args.guidance_scale]),
            dtype=dtype,
        )
    sample, time_ids, steps, timesteps = scheduler.initialize(sample)
    iree_inputs = [
        sample,
        ireert.asdevicearray(unet_runner.config.device, prompt_embeds),
        ireert.asdevicearray(unet_runner.config.device, text_embeds),
        time_ids,
        None,
    ]
    for i in range(steps.to_host()):
        # print(f"step {i}")
        if args.scheduler_vmfb_path:
            step_index = ireert.asdevicearray(
                unet_runner.config.device, torch.tensor([i]), dtype="int64"
            )
        else:
            step_index = i
        latents, t = scheduler.scale_model_input(
            sample,
            step_index,
            timesteps,
        )
        noise_pred = unet_runner.ctx.modules.compiled_unet["run_forward"](
            latents,
            t,
            iree_inputs[1],
            iree_inputs[2],
            iree_inputs[3],
        )
        sample = scheduler.step(
            noise_pred,
            t,
            sample,
            guidance_scale,
            step_index,
        )
    return sample


@torch.no_grad()
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
    scheduler = schedulers.get_scheduler(args.hf_model_name, args.scheduler_id)
    if args.scheduler_id == "PNDM":
        scheduler.config.skip_prk_steps = True
    scheduler.set_timesteps(args.num_inference_steps)
    timesteps = scheduler.timesteps
    print(timesteps)
    sample = sample * scheduler.init_noise_sigma

    height = args.height
    width = args.width
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)

    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=torch.float32)
    add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
    add_time_ids = add_time_ids.repeat(args.batch_size * 1, 1)
    sample = sample.to(torch.float32)
    prompt_embeds = prompt_embeds.to(torch.float32)
    text_embeds = text_embeds.to(torch.float32)

    for idx, t in enumerate(timesteps):
        print(t)
        latent_model_input = torch.cat([sample] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        noise_pred = unet_model.forward(
            latent_model_input,
            t,
            prompt_embeds,
            text_embeds,
            add_time_ids,
        )
        # print("NOISE_PRED: ", noise_pred)
        # print("STEP_INDEX : ", idx)
        noise_preds = noise_pred.chunk(2)
        noise_pred = noise_preds[0] + args.guidance_scale * (
            noise_preds[1] - noise_preds[0]
        )
        sample = scheduler.step(
            noise_pred,
            t,
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
    time_ids = torch.rand(init_batch_dim * args.batch_size, 6, dtype=dtype)
    if args.compiled_pipeline:
        assert (
            args.pipeline_vmfb_path is not None
        ), "--pipeline_vmfb_path is required for compiled pipeline run"
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
        turbine_output = turbine_compiled_output
    elif args.split_scheduler:
        turbine_split_output = run_unet_split_scheduled(
            sample,
            prompt_embeds,
            text_embeds,
            args,
        )
        if args.scheduler_vmfb_path:
            turbine_split_output = turbine_split_output.to_host()
        print(
            "TURBINE SPLIT OUTPUT:",
            turbine_split_output,
            turbine_split_output.shape,
            turbine_split_output.dtype,
        )
        turbine_output = turbine_split_output
    else:
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
        turbine_output = turbine_python_output

    if args.compare_vs_torch:
        if args.scheduler_id == "EulerAncestralDiscrete" and args.scheduler_vmfb_path:
            print(
                f"WARNING: {args.scheduler_id} scheduler adds random noise to results and we haven't piped through a torch generator yet to fix the seed. Expect mismatch results."
            )
        if args.scheduler_id == "PNDM" and args.scheduler_vmfb_path:
            print(
                f"WARNING: {args.scheduler_id} scheduler normally uses data-dependent control flow with counters and other data dependence. Expect different results after 1 step."
            )
        print("generating torch output: ")
        torch_output = run_torch_diffusers_loop(
            sample,
            prompt_embeds,
            text_embeds,
            args,
        )
        print("torch OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)

        print("\n(torch (diffusers) image latents to iree image latents): ")
        try:
            np.testing.assert_allclose(
                turbine_output, torch_output, rtol=4e-2, atol=4e-2
            )
            print("passed!")
        except AssertionError as err:
            if args.scheduler_id == "EulerAncestralDiscrete":
                print(
                    "Expected failure matching numerics due to intentionally random noise in results."
                )
            print(err)
