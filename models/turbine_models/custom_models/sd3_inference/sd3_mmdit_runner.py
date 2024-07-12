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
    args,
):
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
    ]
    noise_pred = mmdit_runner.ctx.modules.compiled_mmdit["run_forward"](
        *iree_inputs
    ).to_host()
    return noise_pred


@torch.no_grad()
def run_diffusers_mmdit(
    hidden_states,
    encoder_hidden_states,
    pooled_projections,
    timestep,
    args,
):
    from turbine_models.custom_models.sd3_inference.sd3_mmdit import MMDiTModel

    mmdit_model = MMDiTModel(
        args.hf_model_name,
        dtype=torch.float32,
    )
    noise_pred = mmdit_model.forward(
        hidden_states.float(),
        encoder_hidden_states.float(),
        pooled_projections.float(),
        timestep.float(),
    )

    return noise_pred.numpy()


def run_attn_turbine(q, k, v, args):
    attn_runner = vmfbRunner(
        args.device,
        args.vmfb_path,
        None,
    )
    iree_inputs = [
        ireert.asdevicearray(attn_runner.config.device, q),
        ireert.asdevicearray(attn_runner.config.device, k),
        ireert.asdevicearray(attn_runner.config.device, v),
    ]
    attn_output = attn_runner.ctx.modules.compiled_attn["run_forward"](
        *iree_inputs
    ).to_host()
    return attn_output


@torch.no_grad()
def run_attn_torch(q, k, v, args):
    from turbine_models.custom_models.sd3_inference.sd3_mmdit import MMDiTAttention

    mmdit_attn = MMDiTAttention()
    attn_output = mmdit_attn.forward(
        torch.tensor(q, dtype=torch.float32),
        torch.tensor(k, dtype=torch.float32),
        torch.tensor(v, dtype=torch.float32),
    )

    return attn_output.numpy()


def find_errs(turbine_output, torch_output, dim=[], failed_dims=[], errs=[]):
    if not np.allclose(turbine_output, torch_output, rtol=4e-2, atol=4e-2):
        if turbine_output.ndim > 0:
            orig_dim = dim
            for idx, i in enumerate(torch_output):
                dim = [*orig_dim, idx]
                try:
                    np.testing.assert_allclose(
                        turbine_output[idx], torch_output[idx], rtol=4e-2, atol=4e-2
                    )
                except Exception as e:
                    err = np.abs(turbine_output[idx] - torch_output[idx])
                    failed_dims.append(dim)
                    errs.append([err, turbine_output[idx], torch_output[idx]])
                    failed_dims, errs = find_errs(
                        turbine_output[idx], torch_output[idx], dim, failed_dims, errs
                    )
    return (failed_dims, errs)


if __name__ == "__main__":
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args
    import numpy as np
    import os

    torch.random.manual_seed(0)

    if args.precision == "fp16":
        dtype = torch.float16
        np_dtype = np.float16
    else:
        dtype = torch.float32
        np_dtype = np.float32

    if args.attn_repro:
        qkv_shape = (2, 24, 4250, 64)
        example_qkv = [
            np.load("q.npy").astype(np_dtype),
            np.load("k.npy").astype(np_dtype),
            np.load("v.npy").astype(np_dtype),
        ]
        turbine_output = run_attn_turbine(
            *example_qkv,
            args,
        )
        torch_output = run_attn_torch(*example_qkv, args).astype(np.float16)
        np.save("turbine_attn_output.npy", turbine_output)
        np.save("torch_attn_output.npy", torch_output)
        failed_dims, errs = find_errs(turbine_output, torch_output)
        for idx, dim in enumerate(failed_dims):
            if len(dim) == len(torch_output.shape):
                print("Failed dimension: ", dim, " with error: ", errs[idx][0])
                print("Turbine output: ", errs[idx][1])
                print("Torch output: ", errs[idx][2])
        print(torch_output.shape)
        exit()

    batch_size = args.batch_size * 2  # do classifier free guidance
    hidden_states = torch.randn(
        (batch_size, 16, args.height // 8, args.width // 8), dtype=dtype
    )
    encoder_hidden_states = torch.randn(
        (batch_size, args.max_length * 2, 4096), dtype=dtype
    )
    pooled_projections = torch.randn((batch_size, 2048), dtype=dtype)
    timestep = torch.tensor([0, 0], dtype=dtype)

    turbine_output = run_mmdit_turbine(
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
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
            args,
        )
        np.save("torch_mmdit_output.npy", torch_output.astype(np.float16))
        print("torch OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)

        print("\n(torch (comfy) image latents to iree image latents): ")

        np.testing.assert_allclose(turbine_output, torch_output, rtol=4e-2, atol=4e-2)
        print("passed!")
