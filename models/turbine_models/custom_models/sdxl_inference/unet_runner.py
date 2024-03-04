import argparse
from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
import torch
import glob
import numpy as np
from tqdm.auto import tqdm

torch.random.manual_seed(0)


def load_tensor_by_pattern(filename_pattern, load_as_numpy=False):
    """Loads a torch tensor from the first file matching the given pattern in the current working directory.

    Args:
        filename_pattern (str): The filename pattern to match.

    Returns:
        torch.Tensor: The loaded tensor.

    Raises:
        RuntimeError: If multiple files match the pattern.
        ValueError: If no files match the pattern.
    """

    matching_filenames = glob.glob(filename_pattern)

    if len(matching_filenames) == 1:
        first_filename = matching_filenames[0]
        if load_as_numpy:
            tensor = np.load(first_filename)
            print(f"Loaded np array from file: {first_filename}")
        else:
            tensor = torch.load(first_filename)
            print(f"Loaded tensor from file: {first_filename}")
        return tensor
    elif len(matching_filenames) > 1:
        raise RuntimeError(f"Multiple files found matching pattern: {filename_pattern}.")
    else:
        raise ValueError(f"No files found matching pattern: {filename_pattern}.")


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
    # results = unet_model.forward(
    #     sample, timestep, prompt_embeds, text_embeds, time_ids, guidance_scale
    # )
    # np_torch_output = results.detach().cpu().numpy()
    np_torch_output= load_tensor_by_pattern('output_*.npy', load_as_numpy=True)
    return np_torch_output


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    if args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    # sample = torch.rand(
    #     args.batch_size, 4, args.height // 8, args.width // 8, dtype=dtype
    # )
    # timestep = torch.zeros(1, dtype=torch.int64)
    # prompt_embeds = torch.rand(2 * args.batch_size, args.max_length, 2048, dtype=dtype)
    # text_embeds = torch.rand(2 * args.batch_size, 1280, dtype=dtype)
    # time_ids = torch.zeros(2 * args.batch_size, 6, dtype=dtype)
    # guidance_scale = torch.tensor([7.5], dtype=dtype)
    sample = load_tensor_by_pattern("sample_*_f16.pt")
    timestep = load_tensor_by_pattern("timestep_*.pt")
    prompt_embeds = load_tensor_by_pattern("promptembeds_*_f16.pt")
    text_embeds = load_tensor_by_pattern("textembeds_*_f16.pt")
    time_ids = load_tensor_by_pattern("timeids_*_f16.pt")
    guidance_scale = load_tensor_by_pattern("guidancescale_*_f16.pt")

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

        # comment out .float for fp16... sorry.
        # torch_output = run_torch_unet(
        #     args.hf_model_name,
        #     args.hf_auth_token,
        #     sample.float(),
        #     timestep,
        #     prompt_embeds.float(),
        #     text_embeds.float(),
        #     time_ids.float(),
        #     guidance_scale.float(),
        #     # precision="fp16",
        # )
        torch_output_f32 = load_tensor_by_pattern('output_*_f32.npy', load_as_numpy=True)
        torch_output_f16 = load_tensor_by_pattern('output_*_f16.npy', load_as_numpy=True)

        # import pdb; pdb.set_trace()
        # print("TORCH fp32 OUTPUT:", torch_output_f32, torch_output_f32.shape, torch_output_f32.dtype)
        err = utils.largest_error(torch_output_f32, turbine_output)
        print("Largest Error between torch f32 vs rocm f16: ", err)


        print("-----------------------------------------------------------")
        print(f"Compare f16 pytorch to f16 rocm")
        # np.testing.assert_allclose(torch_output_f16, turbine_output.to_host(), atol=4e-2, rtol=4e-2)
        is_close = np.isclose(turbine_output.to_host(), torch_output_f16, rtol=4e-2, atol=4e-2)
        pct = is_close.sum()/torch_output_f16.size * 100
        print(f"pct correct : ({is_close.sum()}/{torch_output_f16.size}) ({pct}%)")
        print(f"largest Error : {utils.largest_error(torch_output_f16, turbine_output)}")
        print("-----------------------------------------------------------")
        print(f"Compare f32 pytorch to f16->f32 rocm")
        # np.testing.assert_allclose(torch_output_f32, turbine_output.to_host().astype(np.float32), atol=4e-2, rtol=4e-2)
        is_close = np.isclose(turbine_output.to_host(), torch_output_f32, rtol=4e-2, atol=4e-2)
        pct = is_close.sum()/torch_output_f32.size * 100
        # print(is_close)
        print(f"pct correct : ({is_close.sum()}/{torch_output_f32.size}) ({pct}%)")
        print(f"largest Error : {utils.largest_error(torch_output_f32, turbine_output.to_host().astype(np.float32))}")
        print("-----------------------------------------------------------")
        print(f"Compare f32 pytorch to f16->f32 pytorch")
        is_close = np.isclose(torch_output_f16.astype(np.float32), torch_output_f32, rtol=4e-2, atol=4e-2)
        pct = is_close.sum()/torch_output_f32.size * 100
        print(f"pct correct : ({is_close.sum()}/{torch_output_f32.size}) ({pct}%)")
        print(f"largest Error : {utils.largest_error(torch_output_f32, torch_output_f16.astype(np.float32))}")
        print("-----------------------------------------------------------")

        # assert err < 9e-3

    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output = None
