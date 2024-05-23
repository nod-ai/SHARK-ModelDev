import argparse
from turbine_models.model_runner import vmfbRunner
from transformers import CLIPTokenizer
from iree import runtime as ireert
import torch


def run_unet(
    device,
    sample,
    timestep,
    encoder_hidden_states,
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
        ireert.asdevicearray(runner.config.device, encoder_hidden_states),
        ireert.asdevicearray(runner.config.device, guidance_scale),
    ]
    results = runner.ctx.modules.compiled_unet["main"](*inputs)
    return results


def run_torch_unet(
    hf_model_name,
    hf_auth_token,
    sample,
    timestep,
    encoder_hidden_states,
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
            self.guidance_scale = 7.5

        def forward(self, sample, timestep, encoder_hidden_states, guidance_scale):
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
    results = unet_model.forward(
        sample, timestep, encoder_hidden_states, guidance_scale
    )
    np_torch_output = results.detach().cpu().numpy()
    return np_torch_output


if __name__ == "__main__":
    args = parser.parse_args()
    sample = torch.rand(
        args.batch_size, 4, args.height // 8, args.width // 8, dtype=torch.float32
    )
    timestep = torch.zeros(1, dtype=torch.float32)
    guidance_scale = torch.Tensor([7.5], dtype=torch.float32)
    if args.hf_model_name == "CompVis/stable-diffusion-v1-4":
        encoder_hidden_states = torch.rand(2, 77, 768, dtype=torch.float32)
    elif args.hf_model_name == "stabilityai/stable-diffusion-2-1-base":
        encoder_hidden_states = torch.rand(2, 77, 1024, dtype=torch.float32)

    turbine_output = run_unet(
        args.device,
        sample,
        timestep,
        encoder_hidden_states,
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
        from turbine_models.custom_models.sd_inference.sd_cmd_opts import args

        torch_output = run_torch_unet(
            args.hf_model_name,
            args.hf_auth_token,
            sample,
            timestep,
            encoder_hidden_states,
            guidance_scale,
        )
        print("TORCH OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)
        err = utils.largest_error(torch_output, turbine_output)
        print("Largest Error: ", err)
        assert err < 9e-5

    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output = None
