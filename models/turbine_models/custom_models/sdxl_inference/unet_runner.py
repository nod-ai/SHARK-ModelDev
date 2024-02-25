import argparse
from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
import torch

torch.random.manual_seed(0)

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
parser.add_argument(
    "--precision", type=str, default="fp32", help="Precision of Stable Diffusion"
)
parser.add_argument(
    "--max_length", type=int, default=77, help="Max input length of Stable Diffusion"
)


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
):
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


def run_torch_unet(
    hf_model_name,
    hf_auth_token,
    sample,
    timestep,
    prompt_embeds,
    text_embeds,
    time_ids,
    guidance_scale,
):
    from diffusers import UNet2DConditionModel

    class UnetModel(torch.nn.Module):
        def __init__(self, hf_model_name, hf_auth_token):
            super().__init__()
            self.unet = UNet2DConditionModel.from_pretrained(
                hf_model_name,
                subfolder="unet",
                token=hf_auth_token,
            )

        def forward(
            self,
            sample,
            timestep,
            prompt_embeds,
            text_embeds,
            time_ids,
            guidance_scale,
        ):
            with torch.no_grad():
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": time_ids,
                }
                samples = torch.cat([sample] * 2)
                noise_pred = self.unet.forward(
                    samples,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            return noise_pred

    unet_model = UnetModel(
        hf_model_name,
        hf_auth_token,
    )
    results = unet_model.forward(
        sample, timestep, prompt_embeds, text_embeds, time_ids, guidance_scale
    )
    np_torch_output = results.detach().cpu().numpy()
    return np_torch_output


if __name__ == "__main__":
    args = parser.parse_args()
    if args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    sample = torch.rand(
        args.batch_size, 4, args.height // 8, args.width // 8, dtype=dtype
    )
    timestep = torch.zeros(1, dtype=dtype)
    prompt_embeds = torch.rand(2 * args.batch_size, args.max_length, 2048, dtype=dtype)
    text_embeds = torch.rand(2 * args.batch_size, 1280, dtype=dtype)
    time_ids = torch.zeros(2 * args.batch_size, 6, dtype=dtype)
    guidance_scale = torch.tensor([7.5], dtype=dtype)
    if args.hf_model_name == "CompVis/stable-diffusion-v1-4":
        encoder_hidden_states = torch.rand(2, args.max_length, 768, dtype=dtype)
    elif args.hf_model_name == "stabilityai/stable-diffusion-2-1-base":
        encoder_hidden_states = torch.rand(2, args.max_length, 1024, dtype=dtype)

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
        assert err < 9e-5

    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output = None
