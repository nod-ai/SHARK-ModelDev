import argparse
from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
import torch
import glob
import numpy as np
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()

# TODO move common runner flags to generic flag file
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
parser.add_argument(
    "--precision", type=str, default="fp32", help="Precision of Stable Diffusion"
)
parser.add_argument("--variant", type=str, default="decode")


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


def run_vae(
    device,
    example_input,
    vmfb_path,
    hf_model_name,
    external_weight_path,
):
    runner = vmfbRunner(device, vmfb_path, external_weight_path)
    inputs = [ireert.asdevicearray(runner.config.device, example_input)]
    results = runner.ctx.modules.compiled_vae["main"](*inputs)

    return results


def run_torch_vae(hf_model_name, custom_vae, variant, example_input):
    from diffusers import AutoencoderKL

    class VaeModel(torch.nn.Module):
        def __init__(
            self,
            hf_model_name,
            custom_vae=custom_vae,
        ):
            super().__init__()
            self.vae = None
            if custom_vae in ["", None]:
                self.vae = AutoencoderKL.from_pretrained(
                    hf_model_name,
                    subfolder="vae",
                )
            elif not isinstance(custom_vae, dict):
                try:
                    # custom HF repo with no vae subfolder
                    self.vae = AutoencoderKL.from_pretrained(
                        custom_vae,
                    )
                except:
                    # some larger repo with vae subfolder
                    self.vae = AutoencoderKL.from_pretrained(
                        custom_vae,
                        subfolder="vae",
                    )
            else:
                # custom vae as a HF state dict
                self.vae = AutoencoderKL.from_pretrained(
                    hf_model_name,
                    subfolder="vae",
                )
                self.vae.load_state_dict(custom_vae)

        def decode_inp(self, inp):
            inp = inp / 0.13025
            x = self.vae.decode(inp, return_dict=False)[0]
            return (x / 2 + 0.5).clamp(0, 1)

        def encode_inp(self, inp):
            latents = self.vae.encode(inp).latent_dist.sample()
            return 0.13025 * latents

    vae_model = VaeModel(
        hf_model_name,
    )

    if variant == "decode":
        results = vae_model.decode_inp(example_input)
    elif variant == "encode":
        results = vae_model.encode_inp(example_input)
    np_torch_output = results.detach().cpu().numpy()
    return np_torch_output


if __name__ == "__main__":
    args = parser.parse_args()
    if args.precision == "fp16":
        dtype = torch.float16
        custom_vae = "madebyollin/sdxl-vae-fp16-fix"
    else:
        dtype = torch.float32
        custom_vae = ""
    if args.variant == "decode":
        '''example_input = torch.rand(
            args.batch_size, 4, args.height // 8, args.width // 8, dtype=dtype
        )'''
        example_input = load_tensor_by_pattern("example_input_*_f16.pt")
    elif args.variant == "encode":
        example_input = torch.rand(
            args.batch_size, 3, args.height, args.width, dtype=dtype
        )
    print("generating turbine output:")
    turbine_output = run_vae(
        args.device,
        example_input,
        args.vmfb_path,
        args.hf_model_name,
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

        # torch_output = run_torch_vae(
        #     args.hf_model_name, custom_vae, args.variant, example_input.float()
        # )
        torch_output_f32 = load_tensor_by_pattern('output_*_f32.npy', load_as_numpy=True)
        torch_output_f16 = load_tensor_by_pattern('output_*_f16.npy', load_as_numpy=True)

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

        #assert err < 2e-3

    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output = None
