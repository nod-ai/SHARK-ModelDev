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
    iree_dtype,
):
    runner = vmfbRunner(device, vmfb_path, external_weight_path)
    inputs = [
        ireert.asdevicearray(runner.config.device, sample, dtype=iree_dtype),
        ireert.asdevicearray(runner.config.device, timestep, dtype=iree_dtype),
        ireert.asdevicearray(
            runner.config.device, encoder_hidden_states, dtype=iree_dtype
        ),
        ireert.asdevicearray(runner.config.device, guidance_scale, dtype=iree_dtype),
    ]
    results = runner.ctx.modules.compiled_unet["run_forward"](*inputs)
    return results


def run_torch_unet(
    hf_model_name,
    hf_auth_token,
    sample,
    timestep,
    encoder_hidden_states,
    guidance_scale,
):
    from turbine_models.custom_models.sd_inference.unet import UnetModel

    unet_model = UnetModel(
        hf_model_name,
    )
    results = unet_model.forward(
        sample, timestep, encoder_hidden_states, guidance_scale
    )
    np_torch_output = results.detach().cpu().numpy()
    return np_torch_output


if __name__ == "__main__":
    args = parser.parse_args()
    iree_dtypes = {
        "fp16": "float16",
        "fp32": "float32",
    }
    sample = torch.rand(
        args.batch_size * 2, 4, args.height // 8, args.width // 8, dtype=torch.float32
    )
    timestep = torch.zeros(1, dtype=torch.float32)
    guidance_scale = torch.Tensor([7.5], dtype=torch.float32)
    if args.hf_model_name == "CompVis/stable-diffusion-v1-4":
        encoder_hidden_states = torch.rand(2, args.max_length, 768, dtype=torch.float32)
    elif args.hf_model_name == "stabilityai/stable-diffusion-2-1-base":
        encoder_hidden_states = torch.rand(
            2, args.max_length, 1024, dtype=torch.float32
        )

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
        iree_dtypes[args.precision],
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
