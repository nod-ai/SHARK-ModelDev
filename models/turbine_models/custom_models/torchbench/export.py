# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import gc

from iree.compiler.ir import Context
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

from pytorch.benchmarks.dynamo.common import parse_args
from pytorch.benchmarks.dynamo.torchbench import TorchBenchmarkRunner, setup_torchbench_cwd

torchbench_models_dict = {
    # "BERT_pytorch": {
    #     "dim": 128,
    # },
    # "Background_Matting": {
    #     "dim": 16,
    # },
    "LearningToPaint": {
        "dim": 1024,
    },
    "alexnet": {
        "dim": 1024,
    },
    "dcgan": {
        "dim": 1024,
    },
    "densenet121": {
        "dim": 64,
    },
    "hf_Albert": {
        "dim": 32,
        "buffer_prefix": "albert"
    },
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
    "nvidia_deeprecommender": {
        "dim": 1024,
    },
    "pytorch_unet": {
        "dim": 8,
    },
    "resnet18": {
        "dim": 512,
    },
    "resnet50": {
        "dim": 128,
    },
    "resnet50_32x4d": {
        "dim": 128,
    },
    "shufflenet_v2_x1_0": {
        "dim": 512,
    },
    "squeezenet1_1": {
        "dim": 512,
    },
    "timm_nfnet": {
        "dim": 256,
    },
    "timm_efficientnet": {
        "dim": 128,
    },
    "timm_regnet": {
        "dim": 128,
    },
    "timm_resnest": {
        "dim": 256,
    },
    "timm_vision_transformer": {
        "dim": 256,
    },
    "timm_vovnet": {
        "dim": 128,
    },
    "vgg16": {
        "dim": 128,
    },
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


def get_model_and_inputs(model_id, batch_size, tb_dir, tb_args):
    runner = get_runner(tb_dir, tb_args)
    return runner.load_model(
        "cuda:0",
        model_id,
        batch_size = batch_size,
    )


@torch.no_grad()
def export_torchbench_model(
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
):
    static_dim = torchbench_models_dict[model_id]["dim"]
    dtype = torch.float16 if precision == "fp16" else torch.float32
    np_dtype = "float16" if precision == "fp16" else "float32"
    safe_name = utils.create_safe_name(
        model_id,
        f"_{static_dim}_{precision}",
    )
    if decomp_attn:
        safe_name += "_decomp_attn"

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

    _, model_name, model, forward_args, _ = get_model_and_inputs(model_id, batch_size, tb_dir, tb_args)
    
    for idx, i in enumerate(forward_args.values()):
        np.save(f"input{idx}", i.clone().detach().cpu())
    if dtype == torch.float16:
        model = model.half()
        model.to("cuda:0")

    if not isinstance(forward_args, dict):
        forward_args = [i.type(dtype) for i in forward_args]
    
    mapper = {}
    if (external_weights_dir is not None):
        if not os.path.exists(external_weights_dir):
            os.mkdir(external_weights_dir)
        external_weight_path = os.path.join(external_weights_dir, f"{model_id}_{precision}.irpa")


    decomp_list = [torch.ops.aten.reflection_pad2d]
    if decomp_attn == True:
        decomp_list.extend([
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ])
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
            breakpoint()
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
        return vmfb_path

def run_main(model_id, args, tb_dir, tb_args):
    print(f"exporting {model_id}")
    mod_str = export_torchbench_model(
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
    )
    if args.compile_to in ["torch", "mlir"]:
        safe_name = utils.create_safe_name(
            model_id,
            f"_{static_dim}_{precision}",
        )
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
    gc.collect()

if __name__ == "__main__":
    from turbine_models.custom_models.torchbench.cmd_opts import args, unknown
    tb_dir = setup_torchbench_cwd()
    if args.model_id.lower() == "all":
        for name in torchbench_models_dict.keys():
            run_main(name, args, tb_dir, unknown)
    else:
        run_main(args.model_id, args, tb_dir, unknown)

