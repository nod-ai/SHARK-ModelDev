import argparse
from turbine_models.model_runner import vmfbRunner
from turbine_models.custom_models.sd_inference import utils, schedulers
from iree import runtime as ireert
import torch
import numpy as np
from tqdm.auto import tqdm
from shark_turbine.ops.iree import trace_tensor

torch.random.manual_seed(0)


def run_mmdit_turbine(
    hidden_states,
    encoder_hidden_states,
    pooled_projections,
    timestep,
    lora_scale,
    args,
):
    torch_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    mmdit_runner = vmfbRunner(
        args.device,
        args.vmfb_path,
        args.external_weight_path,
    )
    iree_inputs = [
        ireert.asdevicearray(mmdit_runner.config.device, hidden_states),
        ireert.asdevicearray(mmdit_runner.config.device, encoder_hidden_states),
        ireert.asdevicearray(mmdit_runner.config.device, pooled_projections),
        ireert.asdevicearray(mmdit_runner.config.device, timestep),
        ireert.asdevicearray(mmdit_runner.config.device, lora_scale),
    ]
    noise_pred = mmdit_runner.ctx.modules.compiled_mmdit["run_forward"](*iree_inputs).to_host()
    return noise_pred


@torch.no_grad()
def run_diffusers_mmdit(
    hidden_states,
    encoder_hidden_states,
    pooled_projections,
    timestep,
    lora_scale,
    args,
):
    from turbine_models.custom_models.sd3_inference.turbine_mmdit import MMDiTModel
    mmdit_model = MMDiTModel(
        args.hf_model_name,
        dtype=torch.float32,
    )
    noise_pred = mmdit_model.forward(
        hidden_states.float(), encoder_hidden_states.float(), pooled_projections.float(), timestep.float(), lora_scale.float()
    )

    return noise_pred.numpy()


if __name__ == "__main__":
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args
    import numpy as np
    import os

    torch.random.manual_seed(0)

    if args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    hidden_states = torch.randn(
        (args.batch_size, 16, args.height // 8, args.width // 8), dtype=dtype
    )
    encoder_hidden_states = torch.randn(
        (args.batch_size, args.max_length, 4096), dtype=dtype
    )
    pooled_projections = torch.randn((args.batch_size, 2048), dtype=dtype)
    timestep = torch.tensor([0], dtype=dtype)
    lora_scale = torch.tensor([1.0], dtype=dtype)

    turbine_output = run_mmdit_turbine(
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        lora_scale,
        args,
    )
    print(
        "TURBINE SPLIT OUTPUT:",
        turbine_output,
        turbine_output.shape,
        turbine_output.dtype,
    )
    turbine_output = turbine_output

    if args.compare_vs_torch:
        print("generating torch output: ")
        torch_output = run_diffusers_mmdit(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            lora_scale,
            args,
        )
        print("torch OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)

        print("\n(torch (comfy) image latents to iree image latents): ")

        np.testing.assert_allclose(
            turbine_output, torch_output, rtol=4e-2, atol=4e-2
        )
        print("passed!")

