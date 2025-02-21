# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import os
import sys
import safetensors

from iree import runtime as ireert
from iree.compiler.ir import Context
import numpy as np
from iree.turbine.aot import *
from iree.turbine.transforms.general.add_metadata import AddMetadataPass


from turbine_models.custom_models.sd_inference import utils
import torch
from huggingface_hub import hf_hub_download


class UnetModel(torch.nn.Module):
    def __init__(self, hf_model_name, hf_auth_token=None, precision="fp32"):
        from diffusers import UNet2DConditionModel

        super().__init__()
        if precision == "fp16":
            try:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                    variant="fp16",
                )
            except:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                )
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                hf_model_name,
                subfolder="unet",
                auth_token=hf_auth_token,
                low_cpu_mem_usage=False,
            )
        self.do_classifier_free_guidance = True

    def forward(
        self,
        latent_model_input,
        timestep,
        prompt_embeds,
        text_embeds,
        time_ids,
        guidance_scale,
    ):
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        if self.do_classifier_free_guidance:
            latent_model_input = torch.cat([latent_model_input] * 2)
        noise_pred = self.unet.forward(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        if self.do_classifier_free_guidance:
            noise_preds = noise_pred.chunk(2)
            noise_pred = noise_preds[0] + guidance_scale * (
                noise_preds[1] - noise_preds[0]
            )
        return noise_pred


def get_punet_model(hf_model_name, external_weight_path, quant_paths, precision="i8"):
    from sharktank.models.punet.model import (
        Unet2DConditionModel as sharktank_unet2d,
        ClassifierFreeGuidanceUnetModel as sharktank_CFGPunetModel,
    )
    from sharktank.utils import cli

    if precision == "i8":
        repo_id = "amd-shark/sdxl-quant-models"
        subfolder = "unet/int8"
        revision = "a31d1b1cba96f0da388da348bcaee197a073d451"
    elif precision in ["fp16", "fp32"]:
        repo_id = hf_model_name
        subfolder = "unet"
        revision = "defeb489fe2bb17b77d587924db9e58048a8c140"

    def download(filename):
        return hf_hub_download(
            repo_id=repo_id, subfolder=subfolder, filename=filename, revision=revision
        )

    if quant_paths and quant_paths["config"] and os.path.exists(quant_paths["config"]):
        results = {
            "config.json": quant_paths["config"],
        }
    else:
        results = {
            "config.json": download("config.json"),
        }

    if quant_paths and quant_paths["params"] and os.path.exists(quant_paths["params"]):
        results["params.safetensors"] = quant_paths["params"]
    else:
        results["params.safetensors"] = download("params.safetensors")

    output_dir = os.path.dirname(external_weight_path)

    if precision == "i8":
        if (
            quant_paths
            and quant_paths["quant_params"]
            and os.path.exists(quant_paths["quant_params"])
        ):
            results["quant_params.json"] = quant_paths["quant_params"]
        else:
            results["quant_params.json"] = download("quant_params.json")
        ds_filename = os.path.basename(external_weight_path)
        output_path = os.path.join(output_dir, ds_filename)
        ds = get_punet_dataset(
            results["config.json"],
            results["params.safetensors"],
            output_path,
            results["quant_params.json"],
        )
    else:
        ds_filename = (
            os.path.basename(external_weight_path).split("unet")[0]
            + f"punet_dataset_{precision}.irpa"
        )
        output_path = os.path.join(output_dir, ds_filename)
        ds = get_punet_dataset(
            results["config.json"],
            results["params.safetensors"],
            output_path,
        )

    cond_unet = sharktank_unet2d.from_dataset(ds)
    mdl = sharktank_CFGPunetModel(cond_unet)
    return mdl


def get_punet_dataset(
    config_json_path,
    params_path,
    output_path,
    quant_params_path=None,
):
    from sharktank.models.punet.tools import import_brevitas_dataset

    ds_import_args = [
        f"--config-json={config_json_path}",
        f"--params={params_path}",
        f"--output-irpa-file={output_path}",
    ]
    if quant_params_path:
        ds_import_args.extend([f"--quant-params={quant_params_path}"])
    import_brevitas_dataset.main(ds_import_args)
    return import_brevitas_dataset.Dataset.load(output_path)


@torch.no_grad()
def export_unet_model(
    hf_model_name,
    batch_size,
    height,
    width,
    precision="fp32",
    max_length=77,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target=None,
    ireec_flags=None,
    decomp_attn=False,
    exit_on_vmfb=False,
    pipeline_dir=None,
    attn_spec=None,
    input_mlir=None,
    weights_only=False,
    use_punet=False,
    quant_paths=None,
    add_tk_kernels=False,
    tk_kernels_dir=None,
):
    if use_punet:
        submodel_name = "punet"
    else:
        submodel_name = "unet"
    if not attn_spec:
        if (not decomp_attn) and use_punet:
            attn_spec = "punet"
        elif (not decomp_attn) and "gfx9" in target:
            attn_spec = "mfma"
        elif (not decomp_attn) and "gfx11" in target:
            attn_spec = "wmma"
    safe_name = utils.create_safe_name(
        hf_model_name,
        f"_bs{batch_size}_{max_length}_{height}x{width}_{precision}_{submodel_name}",
    )
    if pipeline_dir:
        safe_name = os.path.join(pipeline_dir, safe_name)

    if decomp_attn == True:
        ireec_flags += ",--iree-opt-aggressively-propagate-transposes=False"

    # Currently, only int8 tk kernels are integrated
    if add_tk_kernels and precision != "i8":
        add_tk_kernels = False

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
            flagset_keywords=["punet"] if use_punet else [],
            add_tk_kernels=add_tk_kernels,
            tk_kernels_dir=tk_kernels_dir,
        )
        return vmfb_path
    elif use_punet:
        unet_model = get_punet_model(
            hf_model_name, external_weight_path, quant_paths, precision
        )
    else:
        unet_model = UnetModel(hf_model_name, hf_auth_token, precision)

    mapper = {}
    np_dtypes = {
        "fp16": "float16",
        "fp32": "float32",
        "i8": "int8",
    }
    torch_dtypes = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "i8": torch.int8,
    }
    dtype = torch_dtypes[precision]
    np_dtype = np_dtypes[precision]

    if precision == "fp16" and not use_punet:
        unet_model = unet_model.half()

    if use_punet:
        dtype = torch.float16

    if not use_punet:
        utils.save_external_weights(
            mapper, unet_model, external_weights, external_weight_path
        )

    if weights_only:
        return external_weight_path

    do_classifier_free_guidance = True
    init_batch_dim = 2 if do_classifier_free_guidance else 1

    sample = [
        batch_size,
        4,
        height // 8,
        width // 8,
    ]

    time_ids_shape = (init_batch_dim * batch_size, 6)
    prompt_embeds_shape = (init_batch_dim * batch_size, max_length, 2048)
    text_embeds_shape = (init_batch_dim * batch_size, 1280)
    example_forward_args = [
        torch.empty(sample, dtype=dtype),
        torch.empty(1, dtype=dtype),
        torch.empty(prompt_embeds_shape, dtype=dtype),
        torch.empty(text_embeds_shape, dtype=dtype),
        torch.empty(time_ids_shape, dtype=dtype),
        torch.tensor([7.5], dtype=dtype),
    ]
    example_forward_args_dict = {
        "sample": torch.rand(sample, dtype=dtype),
        "timestep": torch.zeros(1, dtype=dtype),
        "encoder_hidden_states": torch.rand(prompt_embeds_shape, dtype=dtype),
        "text_embeds": torch.rand(text_embeds_shape, dtype=dtype),
        "time_ids": torch.zeros(time_ids_shape, dtype=dtype),
        "guidance_scale": torch.tensor([7.5], dtype=dtype),
    }
    decomp_list = []
    if decomp_attn == True:
        decomp_list = [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ]
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):
        if use_punet:
            output = export(
                unet_model,
                kwargs=example_forward_args_dict,
                module_name="compiled_punet",
            )
            module = output.mlir_module
        else:
            if external_weights:
                externalize_module_parameters(unet_model)
            fxb = FxProgramsBuilder(unet_model)

            @fxb.export_program(
                args=(example_forward_args,),
            )
            def _forward(
                module,
                inputs,
            ):
                return module.forward(*inputs)

            class CompiledUnet(CompiledModule):
                run_forward = _forward

            inst = CompiledUnet(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)

    model_metadata_run_forward = {
        "model_name": "sd_unet",
        "input_shapes": [
            sample,
            (1,),
            prompt_embeds_shape,
            text_embeds_shape,
            time_ids_shape,
            (1,),
        ],
        "input_dtypes": [np_dtype for x in range(6)],
        "output_shapes": [sample],
        "output_dtypes": [np_dtype],
    }

    module = AddMetadataPass(module, model_metadata_run_forward, "run_forward").run()

    module_str = str(module)
    if compile_to != "vmfb":
        return module_str
    else:
        vmfb_path = utils.compile_to_vmfb(
            module_str,
            device,
            target,
            ireec_flags,
            safe_name,
            return_path=True,
            attn_spec=attn_spec,
            flagset_keywords=["punet"] if use_punet else [],
            add_tk_kernels=add_tk_kernels,
            batch_size=batch_size,
            tk_kernels_dir=tk_kernels_dir,
        )
        if exit_on_vmfb:
            exit()
    return vmfb_path


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    if args.input_mlir:
        unet_model = None
    else:
        unet_model = UnetModel(
            args.hf_model_name,
            args.hf_auth_token,
            args.precision,
        )
    mod_str = export_unet_model(
        unet_model,
        args.hf_model_name,
        args.batch_size,
        args.height,
        args.width,
        args.precision,
        args.max_length,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.ireec_flags + args.attn_flags + args.unet_flags,
        args.decomp_attn,
        attn_spec=args.attn_spec,
        input_mlir=args.input_mlir,
        add_tk_kernels=args.add_tk_kernels,
        tk_kernels_dir=args.tk_kernels_dir
    )
    if args.input_mlir:
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name,
        f"_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}_{'p' if args.use_i8_punet else ''}unet",
    )
    if args.compile_to != "vmfb":
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
