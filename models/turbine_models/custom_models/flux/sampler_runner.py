import argparse
from turbine_models.model_runner import vmfbRunner
from turbine_models.custom_models.sd_inference import utils, schedulers
from iree import runtime as ireert
import torch
import numpy as np
from tqdm.auto import tqdm
from shark_turbine.ops.iree import trace_tensor

torch.random.manual_seed(0)


def run_sampler_turbine(
    img,
    img_ids,
    txt,
    txt_ids,
    vec,
    t_curr,
    t_prev,
    guidance_vec,
    args,
):
    sampler_runner = vmfbRunner(
        args.device,
        args.vmfb_path,
        args.external_weight_path,
    )
    inputs = [
        img,
        img_ids,
        txt,
        txt_ids,
        vec,
        t_curr,
        t_prev,
        guidance_vec,
    ]
    iree_inputs = [ireert.asdevicearray(sampler_runner.config.device, i) for i in inputs]
    noise_pred = sampler_runner.ctx.modules.compiled_flux_sampler["run_forward"](
        *iree_inputs
    ).to_host()
    return noise_pred


@torch.no_grad()
def run_sampler_torch(
    img,
    img_ids,
    txt,
    txt_ids,
    vec,
    t_curr,
    t_prev,
    guidance_vec,
    args,
):
    from turbine_models.custom_models.flux.sampler import FluxModel

    sampler_model = FluxModel()
    noise_pred = sampler_model.forward(
        img.type(torch.bfloat16),
        img_ids.type(torch.bfloat16),
        txt.type(torch.bfloat16),
        txt_ids.type(torch.bfloat16),
        vec.type(torch.bfloat16),
        t_curr.type(torch.bfloat16),
        t_prev.type(torch.bfloat16),
        guidance_vec.type(torch.bfloat16),
    )

    return noise_pred.clone().float().cpu().numpy()


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
    attn_output = attn_runner.ctx.modules.compiled_attn["main"](
        *iree_inputs
    ).to_host()
    return attn_output


@torch.no_grad()
def run_attn_torch(q, k, v, args):
    from turbine_models.custom_models.flux.sampler import FluxAttention

    flux_attn = FluxAttention()
    attn_output = flux_attn.forward(
        q,
        k,
        v,
    )

    return attn_output.clone().float().cpu().numpy()


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
    batch_size = args.batch_size
    if args.attn_repro:

        # example_args = [
        #     torch.rand((batch_size, 4096, 3072), dtype=dtype),
        #     torch.rand((batch_size, 512, 3072), dtype=dtype),
        #     torch.rand((batch_size, 3072), dtype=dtype),
        #     torch.rand((batch_size, 4608, 3), dtype=dtype),
        # ]
        example_args = [
            torch.rand((batch_size, 24, 4608, 128), dtype=dtype),
            torch.rand((batch_size, 24, 4608, 128), dtype=dtype),
            torch.rand((batch_size, 24, 4608, 128), dtype=dtype),
        ]
        turbine_output = run_attn_turbine(
            *example_args,
            args,
        )
        torch_output = run_attn_torch(*example_args, args).astype(np.float16)
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
    model_max_len = 64
    img_shape = (
        batch_size,
        int(args.height * args.width / 256),
        64,
    )
    img_ids_shape = (
        batch_size,
        int(args.height * args.width / 256),
        3,
    )
    txt_shape = (
        batch_size,
        model_max_len,
        4096,
    )
    txt_ids_shape = (
        batch_size,
        model_max_len,
        3,
    )
    y_shape = (
        batch_size,
        768,
    )
    example_forward_args = [
        torch.rand(img_shape, dtype=dtype),
        torch.rand(img_ids_shape, dtype=dtype),
        torch.rand(txt_shape, dtype=dtype),
        torch.rand(txt_ids_shape, dtype=dtype),
        torch.rand(y_shape, dtype=dtype),
        torch.rand(1, dtype=dtype),
        torch.rand(1, dtype=dtype),
        torch.rand(batch_size, dtype=dtype),
    ]
    turbine_output = run_sampler_turbine(
        *example_forward_args,
        args,
    )
    print(
        "TURBINE OUTPUT:",
        turbine_output,
        turbine_output.shape,
        turbine_output.dtype,
    )
    turbine_output = turbine_output

    if args.compare_vs_torch:
        print("generating torch output: ")
        torch_output = run_sampler_torch(
            *example_forward_args,
            args,
        )
        np.save("torch_mmdit_output.npy", torch_output.astype(np.float16))
        print("torch OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)

        print("\n(torch (comfy) image latents to iree image latents): ")

        np.testing.assert_allclose(turbine_output, torch_output, rtol=4e-2, atol=4e-2)
        print("passed!")
