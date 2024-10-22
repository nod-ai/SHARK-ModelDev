# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import os
import sys

from iree import runtime as ireert
from iree.compiler.ir import Context
import numpy as np
from iree.turbine.aot import *
from iree.turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from diffusers import AutoencoderKL


class VaeModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            hf_model_name,
            subfolder="vae",
        )

    def decode(self, inp):
        inp = (inp / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(inp, return_dict=False)[0]
        image = image.float()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        return image

    def encode(self, inp):
        image_np = inp / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images)
        image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch
        latent = self.vae.encode(image_torch)
        return latent


def export_vae_model(
    vae_model,
    hf_model_name,
    batch_size,
    height,
    width,
    precision,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target_triple=None,
    ireec_flags=None,
    decomp_attn=False,
    exit_on_vmfb=False,
    pipeline_dir=None,
    attn_spec=None,
    input_mlir=None,
    weights_only=False,
):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    safe_name = utils.create_safe_name(
        hf_model_name,
        f"_bs{batch_size}_{height}x{width}_{precision}_vae",
    )
    if pipeline_dir:
        safe_name = os.path.join(pipeline_dir, safe_name)

    if input_mlir:
        vmfb_path = utils.compile_to_vmfb(
            input_mlir,
            device,
            target_triple,
            ireec_flags,
            safe_name,
            mlir_source="file",
            return_path=not exit_on_vmfb,
            attn_spec=attn_spec,
        )
        return vmfb_path

    if device == "cpu":
        decomp_attn = True

    if dtype == torch.float16:
        vae_model = vae_model.half()
    mapper = {}
    utils.save_external_weights(
        mapper, vae_model, external_weights, external_weight_path
    )
    if weights_only:
        return external_weight_path

    input_image_shape = (height, width, 3)
    input_latents_shape = (batch_size, 16, height // 8, width // 8)
    encode_args = [
        torch.empty(
            input_image_shape,
            dtype=torch.float32,
        )
    ]
    decode_args = [
        torch.empty(
            input_latents_shape,
            dtype=dtype,
        )
    ]
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
        fxb = FxProgramsBuilder(vae_model)

        # @fxb.export_program(args=(encode_args,))
        # def _encode(module, inputs,):
        #     return module.encode(*inputs)

        @fxb.export_program(args=(decode_args,))
        def _decode(module, inputs):
            return module.decode(*inputs)

        class CompiledVae(CompiledModule):
            decode = _decode

        if external_weights:
            externalize_module_parameters(vae_model)

        inst = CompiledVae(context=Context(), import_to="IMPORT")

        module_str = str(CompiledModule.get_mlir_module(inst))

    if compile_to != "vmfb":
        return module_str
    else:
        vmfb_path = utils.compile_to_vmfb(
            module_str,
            device,
            target_triple,
            ireec_flags,
            safe_name,
            return_path=not exit_on_vmfb,
            attn_spec=attn_spec,
        )
        return vmfb_path


if __name__ == "__main__":
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args

    if args.input_mlir:
        vae_model = None
    else:
        vae_model = VaeModel(
            args.hf_model_name,
        )
    mod_str = export_vae_model(
        vae_model,
        args.hf_model_name,
        args.batch_size,
        height=args.height,
        width=args.width,
        precision=args.precision,
        compile_to=args.compile_to,
        external_weights=args.external_weights,
        external_weight_path=args.external_weight_path,
        device=args.device,
        target_triple=args.iree_target_triple,
        ireec_flags=args.ireec_flags + args.attn_flags + args.vae_flags,
        decomp_attn=args.decomp_attn,
        attn_spec=args.attn_spec,
        input_mlir=args.input_mlir,
    )
    if args.input_mlir or (args.compile_to == "vmfb"):
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name,
        f"_bs{str(args.batch_size)}_{args.height}x{args.width}_{args.precision}_vae",
    )
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
