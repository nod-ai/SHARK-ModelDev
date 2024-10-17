# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import gc
import time

from iree.compiler.ir import Context
from iree import runtime as ireert
import numpy as np
from shark_turbine.aot import *
from shark_turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
from turbine_models.custom_models.torchbench import utils
import torch
import torch._dynamo as dynamo
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import argparse
from turbine_models.turbine_tank import turbine_tank
from turbine_models.model_runner import vmfbRunner

from pytorch.benchmarks.dynamo.common import parse_args
from pytorch.benchmarks.dynamo.torchbench import (
    TorchBenchmarkRunner,
    setup_torchbench_cwd,
)

import csv

torchbench_models_all = {
    # "BERT_pytorch": {
    #     "dim": 128,
    # }, # Dynamo Export Issue
    # "Background_Matting": {
    #     "dim": 16,
    # }, # Transpose Bubbling Pattern Failed
    "LearningToPaint": {
        "dim": 1024,
    },
    "alexnet": {
        "dim": 1024,
    },
    "densenet121": {
        "dim": 64,
    },
    # "hf_Albert": {"dim": 32, "buffer_prefix": "albert"},
    # "hf_Bart": {
    #     "dim": 16,
    # },
    # "hf_Bert": {
    #     "dim": 16,
    #     "buffer_prefix": "bert"
    # },
    # "hf_GPT2": {
    #     "dim": 16,
    #     "buffer_prefix": "gpt2"
    # },
    # "hf_T5": {
    #     "dim": 4,
    #     "buffer_prefix": "t5"
    # },
    "mnasnet1_0": {
        "dim": 256,
    },
    "mobilenet_v2": {
        "dim": 128,
    },
    "mobilenet_v3_large": {
        "dim": 256,
    },
    # "nvidia_deeprecommender": {
    #     "dim": 1024,
    # },
    "pytorch_unet": {
        "dim": 8,
    },
    "resnet18": {
        "dim": 512,
    },
    "resnet50": {
        "dim": 128,
    },
    "resnext50_32x4d": {
        "dim": 128,
    },
    "shufflenet_v2_x1_0": {
        "dim": 512,
    },
    "squeezenet1_1": {
        "dim": 512,
    },
    # "timm_nfnet": {
    #     "dim": 256,
    # },
    "timm_efficientnet": {
        "dim": 128,
    },
    "timm_regnet": {
        "dim": 128,
    },
    "timm_resnest": {
        "dim": 256,
    },
    # "timm_vision_transformer": {
    #     "dim": 256,
    #     "decomp_attn": True,
    # },
    "timm_vovnet": {
        "dim": 128,
    },
    # "vgg16": {
    #     "dim": 128,
    # },
}


# Adapted from pytorch.benchmarks.dynamo.common.main()
def get_runner(tb_dir, tb_args):
    if tb_dir:
        os.chdir(tb_dir)
    runner = TorchBenchmarkRunner()
    runner.args = parse_args(tb_args)
    runner.setup_amp()
    runner.model_iter_fn = runner.forward_pass
    return runner


def get_model_and_inputs(model_id, batch_size, tb_dir, tb_args, get_baseline=False):
    runner = get_runner(tb_dir, tb_args)
    _, model_name, model, forward_args, _ = runner.load_model(
        "cuda:0",
        model_id,
        batch_size=batch_size,
    )
    match get_baseline:
        case True:
            start_t = time.time()
            res = runner.forward_pass(model, forward_args, collect_outputs=True)
            baseline = time.time() - start_t
            return model_name, model, forward_args, res, baseline
        case False:
            return model_name, model, forward_args


"""
Imports models from torchbench model tooling, exports them with turbine AOT, and does simple benchmarking.
"""


@torch.no_grad()
def benchmark_torchbench_model(
    model_id,
    tb_dir,
    tb_args,
    precision,
    batch_size=1,
    compile_to="vmfb",
    external_weights=None,
    external_weights_dir=None,
    device=None,
    target=None,
    ireec_flags=None,
    decomp_attn=False,
    exit_on_vmfb=False,
    attn_spec=None,
    input_mlir=None,
    weights_only=False,
    upload_ir=False,
    compare_vs_eager=False,
):
    static_dim = torchbench_models_dict[model_id]["dim"]
    dtype = torch.float16 if precision == "fp16" else torch.float32
    np_dtype = "float16" if precision == "fp16" else "float32"
    safe_name = utils.create_safe_name(
        model_id,
        f"_{static_dim}_{precision}",
    )
    safe_name = os.path.join("generated", safe_name)
    if decomp_attn:
        safe_name += "_decomp_attn"

    if not os.path.exists("generated"):
        os.mkdir("generated")

    if input_mlir:
        vmfb_path = utils.compile_to_vmfb(
            input_mlir,
            device,
            target,
            ireec_flags,
            safe_name,
            mlir_source="file",
            return_path=not exit_on_vmfb,
            attn_spec=attn_spec,
        )
        return vmfb_path

    if compare_vs_eager:
        model_name, model, forward_args, golden, baseline = get_model_and_inputs(
            model_id, batch_size, tb_dir, tb_args, get_baseline=True
        )
    else:
        model_name, model, forward_args = get_model_and_inputs(
            model_id, batch_size, tb_dir, tb_args
        )
        golden = None
        baseline = None

    if dtype == torch.float16:
        model = model.half()
        model.to("cuda:0")

    if not isinstance(forward_args, dict):
        forward_args = [i.type(dtype) for i in forward_args]
        for idx, i in enumerate(forward_args):
            np.save(
                os.path.join("generated", f"{model_id}_input{idx}"),
                i.clone().detach().cpu(),
            )
    else:
        for idx, i in enumerate(forward_args.values()):
            np.save(f"{model_id}_input{idx}", i.clone().detach().cpu())

    mapper = {}
    if external_weights_dir is not None:
        if not os.path.exists(external_weights_dir):
            os.mkdir(external_weights_dir)
        external_weight_path = os.path.join(
            external_weights_dir, f"{model_id}_{precision}.irpa"
        )
    else:
        external_weight_path = None

    decomp_list = [torch.ops.aten.reflection_pad2d]
    if decomp_attn == True or torchbench_models_dict[model_id].get("decomp_attn"):
        print("decomposing attention for: " + model_id)
        decomp_list.extend(
            [
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
                torch.ops.aten._scaled_dot_product_flash_attention.default,
                torch.ops.aten._scaled_dot_product_flash_attention,
                torch.ops.aten.scaled_dot_product_attention,
            ]
        )
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):
        if "hf" in model_id:

            class HF_M(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.mod = model

                def forward(self, inp):
                    return self.mod(**inp)

            if "Bart" not in model_id:
                # In some transformers models, the position ids buffer is registered as non-persistent,
                # which makes it fail to globalize in the FX import.
                # Add them manually to the state dict here.

                prefix = torchbench_models_dict[model_id]["buffer_prefix"]
                getattr(model, prefix).embeddings.register_buffer(
                    "position_ids",
                    getattr(model, prefix).embeddings.position_ids,
                    persistent=True,
                )
            fxb = FxProgramsBuilder(HF_M(model))

            @fxb.export_program(args=(forward_args,))
            def _forward(module: HF_M(model), inputs):
                return module(inputs)

        else:
            fxb = FxProgramsBuilder(model)

            @fxb.export_program(args=(forward_args,))
            def _forward(module, inputs):
                return module(*inputs)

        class CompiledTorchbenchModel(CompiledModule):
            main = _forward

        if external_weights:
            externalize_module_parameters(model)
            save_module_parameters(external_weight_path, model)

        inst = CompiledTorchbenchModel(context=Context(), import_to="IMPORT")

        module = CompiledModule.get_mlir_module(inst)
    model.to("cpu")
    del model
    if compile_to != "vmfb":
        return str(module)
    else:
        vmfb_path = utils.compile_to_vmfb(
            str(module),
            device,
            target,
            ireec_flags,
            safe_name,
            return_path=not exit_on_vmfb,
            attn_spec=attn_spec,
        )
        return vmfb_path, external_weight_path, forward_args, golden, baseline


def _run_iter(runner, inputs):
    start = time.time()
    res = runner.ctx.modules.compiled_torchbench_model["main"](*inputs)
    return res, time.time() - start


def do_compare(shark_results, shark_latency, golden_results, golden_latency):
    numerics_pass_fail = np.allclose(
        shark_results.to_host(),
        golden_results.clone().cpu().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    speedup = golden_latency / shark_latency
    return speedup, numerics_pass_fail


def run_benchmark(
    device,
    vmfb_path,
    weights_path,
    example_args,
    model_id,
    csv_path,
    iters,
    golden=None,
    baseline=None,
):
    if "rocm" in device:
        device = "hip" + device.split("rocm")[-1]
    mod_runner = vmfbRunner(device, vmfb_path, weights_path)
    inputs = torch_to_iree(mod_runner, example_args)
    iter_latencies = []
    for i in range(iters):
        results, iter_latency = _run_iter(mod_runner, inputs)
        iter_latencies.append(iter_latency)
    avg_latency = sum(iter_latencies) / len(iter_latencies)
    it_per_sec = 1 / avg_latency

    if golden is not None and baseline is not None:
        speedup, numerics_pass_fail = do_compare(results, avg_latency, golden, baseline)
    else:
        speedup, numerics_pass_fail = ("N/A", "N/A")

    needs_header = True
    if os.path.exists(csv_path):
        needs_header = False
    with open(csv_path, "a") as csvfile:
        fieldnames = [
            "model",
            "avg_latency",
            "avg_iter_per_sec",
            "speedup_over_eager",
            "numerics",
        ]
        data = [
            {
                "model": model_id,
                "avg_latency": avg_latency,
                "avg_iter_per_sec": it_per_sec,
                "speedup_over_eager": speedup,
                "numerics": numerics_pass_fail,
            }
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(data)
    print(data)


def torch_to_iree(iree_runner, example_args):
    if isinstance(example_args, dict):
        iree_args = [
            ireert.asdevicearray(iree_runner.config.device, i.clone().detach().cpu())
            for i in example_args.values()
        ]
    else:
        iree_args = [
            ireert.asdevicearray(iree_runner.config.device, i.clone().detach().cpu())
            for i in example_args
        ]
    return iree_args


def run_main(model_id, args, tb_dir, tb_args):
    print(f"exporting {model_id}")
    mod_str, weights_path, example_args, golden, baseline = benchmark_torchbench_model(
        model_id,
        tb_dir,
        tb_args,
        precision=args.precision,
        batch_size=args.batch_size,
        compile_to=args.compile_to,
        external_weights=args.external_weights,
        external_weights_dir=args.external_weights_dir,
        device=args.device,
        target=args.target,
        ireec_flags=args.ireec_flags,
        decomp_attn=args.decomp_attn,
        attn_spec=args.attn_spec,
        input_mlir=args.input_mlir,
        compare_vs_eager=args.compare_vs_torch,
    )
    if args.compile_to in ["torch", "mlir"]:
        safe_name = utils.create_safe_name(
            model_id,
            f"_{static_dim}_{precision}",
        )
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
    elif args.run_benchmark:
        run_benchmark(
            args.device,
            mod_str,
            weights_path,
            example_args,
            model_id,
            args.output_csv,
            args.num_iters,
            golden,
            baseline,
        )

    gc.collect()


if __name__ == "__main__":
    from turbine_models.custom_models.torchbench.cmd_opts import args, unknown
    import json

    torchbench_models_dict = json.load(args.model_list_json)
    for list in args.model_lists:
        torchbench_models_dict = json.load(list)
        with open(args.models_json, "r") as f:
            torchbench_models_dict = json.load(file)

        tb_dir = setup_torchbench_cwd()
        if args.model_id.lower() == "all":
            for name in torchbench_models_dict.keys():
                run_main(name, args, tb_dir, unknown)
        else:
            run_main(args.model_id, args, tb_dir, unknown)
