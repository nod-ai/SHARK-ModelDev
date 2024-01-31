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
    "--height", type=int, default=512, help="Height of Stable Diffusion"
)
parser.add_argument("--width", type=int, default=512, help="Width of Stable Diffusion")


def run_scheduler(
    device,
    sample,
    encoder_hidden_states,
    vmfb_path,
    hf_model_name,
    hf_auth_token,
    external_weight_path,
):
    runner = vmfbRunner(device, vmfb_path, external_weight_path)

    inputs = [
        ireert.asdevicearray(runner.config.device, sample),
        ireert.asdevicearray(runner.config.device, encoder_hidden_states),
    ]
    results = runner.ctx.modules.compiled_scheduler["main"](*inputs)
    return results

'''
def run_torch_unet(
    hf_model_name, hf_auth_token, sample, timestep, encoder_hidden_states
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
            self.guidance_scale = 7.5

        def forward(self, sample, timestep, encoder_hidden_states):
            samples = torch.cat([sample] * 2)
            unet_out = self.unet.forward(
                samples, timestep, encoder_hidden_states, return_dict=False
            )[0]
            noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            return noise_pred

    unet_model = UnetModel(
        hf_model_name,
        hf_auth_token,
    )
    results = unet_model.forward(sample, timestep, encoder_hidden_states)
    np_torch_output = results.detach().cpu().numpy()
    return np_torch_output
'''

if __name__ == "__main__":
    args = parser.parse_args()
    sample = torch.rand(
        args.batch_size, 4, args.height // 8, args.width // 8, dtype=torch.float32
    )
    if args.hf_model_name == "CompVis/stable-diffusion-v1-4":
        encoder_hidden_states = torch.rand(2, 77, 768, dtype=torch.float32)
    elif args.hf_model_name == "stabilityai/stable-diffusion-2-1-base":
        encoder_hidden_states = torch.rand(2, 77, 1024, dtype=torch.float32)

    turbine_output = run_scheduler(
        args.device,
        sample,
        encoder_hidden_states,
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


    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output = None
