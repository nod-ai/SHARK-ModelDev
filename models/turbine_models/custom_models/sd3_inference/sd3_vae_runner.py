import argparse
from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
import torch

torch.random.manual_seed(0)


def run_vae(
    device,
    example_input,
    vmfb_path,
    hf_model_name,
    external_weight_path,
):
    runner = vmfbRunner(device, vmfb_path, external_weight_path)
    inputs = [ireert.asdevicearray(runner.config.device, example_input)]
    results = runner.ctx.modules.compiled_vae["decode"](*inputs).to_host()
    results = imagearray_from_vae_out(results)
    return results


def run_torch_vae(hf_model_name, variant, example_input):
    from turbine_models.custom_models.sd3_inference.sd3_vae import VaeModel

    vae_model = VaeModel(
        hf_model_name,
    )

    if variant == "decode":
        results = vae_model.decode(example_input)
    elif variant == "encode":
        results = vae_model.encode(example_input)
    np_torch_output = results.detach().cpu().numpy()
    np_torch_output = imagearray_from_vae_out(np_torch_output)
    return np_torch_output


def imagearray_from_vae_out(image):
    if image.ndim == 4:
        image = image[0]
    image = torch.from_numpy(image).cpu().permute(1, 2, 0).float().numpy()
    image = (image * 255).round().astype("uint8")
    return image


if __name__ == "__main__":
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args
    import numpy as np
    from PIL import Image

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    if args.vae_variant == "decode":
        example_input = torch.rand(
            args.batch_size, 16, args.height // 8, args.width // 8, dtype=dtype
        )
        if args.vae_input_path:
            example_input = np.load(args.vae_input_path)
            if example_input.shape[0] == 2:
                example_input = np.split(example_input, 2)[0]
    elif args.vae_variant == "encode":
        example_input = torch.rand(
            args.batch_size, 3, args.height, args.width, dtype=dtype
        )
    print("generating turbine output:")
    turbine_results = run_vae(
        args.device,
        example_input,
        args.vmfb_path,
        args.hf_model_name,
        args.external_weight_path,
    )
    print(
        "TURBINE OUTPUT:",
        turbine_results,
        turbine_results.shape,
        turbine_results.dtype,
    )
    if args.compare_vs_torch:
        print("generating torch output: ")
        from turbine_models.custom_models.sd_inference import utils

        torch_output = run_torch_vae(
            args.hf_model_name, args.vae_variant, torch.tensor(example_input).float()
        )
        print("TORCH OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)
        if args.vae_input_path:
            out_image_torch = Image.fromarray(torch_output)
            out_image_torch.save("vae_test_output_torch.png")
            out_image_turbine = Image.fromarray(turbine_results)
            out_image_turbine.save("vae_test_output_turbine.png")
        # Allow a small amount of wiggle room for rounding errors (1)

        np.testing.assert_allclose(turbine_results, torch_output, rtol=1, atol=1)
