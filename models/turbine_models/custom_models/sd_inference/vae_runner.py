import argparse
from turbine_models.model_runner import vmfbRunner
from transformers import CLIPTokenizer
from iree import runtime as ireert
import torch

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
    default="CompVis/stable-diffusion-v1-4",
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
    "--height", type=int, default=512, help="Height of Stable Diffusion"
)
parser.add_argument("--width", type=int, default=512, help="Width of Stable Diffusion")
parser.add_argument("--variant", type=str, default="decode")


def run_vae(device, example_input, vmfb_path, hf_model_name, external_weight_path):
    runner = vmfbRunner(device, vmfb_path, external_weight_path)

    inputs = [ireert.asdevicearray(runner.config.device, example_input)]

    results = runner.ctx.modules.compiled_vae["main"](*inputs).to_host()

    return results


def run_torch_vae(hf_model_name, variant, example_input):
    from diffusers import AutoencoderKL

    class VaeModel(torch.nn.Module):
        def __init__(
            self,
            hf_model_name,
            base_vae=False,
            custom_vae="",
            low_cpu_mem_usage=False,
            hf_auth_token="",
        ):
            super().__init__()
            self.vae = None
            if custom_vae == "":
                self.vae = AutoencoderKL.from_pretrained(
                    hf_model_name,
                    subfolder="vae",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    hf_auth_token=hf_auth_token,
                )
            elif not isinstance(custom_vae, dict):
                self.vae = AutoencoderKL.from_pretrained(
                    custom_vae,
                    subfolder="vae",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    hf_auth_token=hf_auth_token,
                )
            else:
                self.vae = AutoencoderKL.from_pretrained(
                    hf_model_name,
                    subfolder="vae",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    hf_auth_token=hf_auth_token,
                )
                self.vae.load_state_dict(custom_vae)
            self.base_vae = base_vae

        def decode_inp(self, input):
            with torch.no_grad():
                input = 1 / 0.18215 * input
                x = self.vae.decode(input, return_dict=False)[0]
            return (x / 2 + 0.5).clamp(0, 1)

        def encode_inp(self, inp):
            latents = self.vae.encode(inp).latent_dist.sample()
            return 0.18215 * latents

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
    if args.variant == "decode":
        example_input = torch.rand(
            args.batch_size, 4, args.height // 8, args.width // 8, dtype=torch.float32
        )
    elif args.variant == "encode":
        example_input = torch.rand(
            args.batch_size, 3, args.height, args.width, dtype=torch.float32
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
        turbine_results.to_host(),
        turbine_results.to_host().shape,
        turbine_results.to_host().dtype,
    )
    if args.compare_vs_torch:
        print("generating torch output: ")
        from turbine_models.custom_models.sd_inference import utils

        torch_output = run_torch_vae(
            args.hf_model_name, args.hf_auth_token, args.variant, example_input
        )
        print("TORCH OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)
        err = utils.largest_error(torch_output, turbine_results)
        print("Largest Error: ", err)
        assert err < 3e-3

    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_results = None
