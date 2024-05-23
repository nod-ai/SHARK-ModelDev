# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree import runtime as ireert
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from shark_turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from diffusers import AutoencoderKL
import argparse
from turbine_models.turbine_tank import turbine_tank

class VaeModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        custom_vae="",
    ):
        super().__init__()
        self.vae = None
        if custom_vae in ["", None]:
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
        elif not isinstance(custom_vae, dict):
            try:
                # custom HF repo with no vae subfolder
                self.vae = AutoencoderKL.from_pretrained(
                    custom_vae,
                )
            except:
                # some larger repo with vae subfolder
                self.vae = AutoencoderKL.from_pretrained(
                    custom_vae,
                    subfolder="vae",
                )
        else:
            # custom vae as a HF state dict
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
            self.vae.load_state_dict(custom_vae)

    def decode_inp(self, inp):
        inp = 1 / 0.18215 * inp
        x = self.vae.decode(inp, return_dict=False)[0]
        return (x / 2 + 0.5).clamp(0, 1)

    def encode_inp(self, inp):
        latents = self.vae.encode(inp).latent_dist.sample()
        return 0.18215 * latents


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
    variant="decode",
    decomp_attn=False,
    exit_on_vmfb=False,
    pipeline_dir=None,
    attn_spec=None,
    input_mlir=None,
    weights_only=False,
):
    if pipeline_dir:
        safe_name = os.path.join(pipeline_dir, "vae_" + variant)
    else:
        safe_name = utils.create_safe_name(
            hf_model_name, f"_bs{batch_size}_{height}x{width}_{precision}_vae_{variant}_{device}"
        )
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
    mapper = {}
    decomp_list = DEFAULT_DECOMPOSITIONS
    if decomp_attn:
        decomp_list.extend(
            [
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
                torch.ops.aten._scaled_dot_product_flash_attention.default,
            ]
        )
    dtype = torch.float16 if precision == "fp16" else torch.float32
    vae_model = vae_model.to(dtype)
    utils.save_external_weights(
        mapper, vae_model, external_weights, external_weight_path
    )
    if weights_only:
        return external_weight_path
    sample = (batch_size, 4, height // 8, width // 8)
    if variant == "encode":
        sample = (batch_size, 3, height, width)

    class CompiledVae(CompiledModule):
        params = export_parameters(vae_model)

        def main(self, inp=AbstractTensor(*sample, dtype=dtype)):
            if variant == "decode":
                return jittable(vae_model.decode_inp, decompose_ops=decomp_list)(inp)
            elif variant == "encode":
                return jittable(vae_model.encode_inp, decompose_ops=decomp_list)(inp)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledVae(context=Context(), import_to=import_to)

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
    from turbine_models.custom_models.sd_inference.sd_cmd_opts import args
    if args.input_mlir:
        vae_model = None
    else:
        vae_model = VaeModel(
            args.hf_model_name,
            custom_vae=custom_vae,
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
        variant=args.vae_variant,
        decomp_attn=args.decomp_attn,
        attn_spec=args.attn_spec,
        input_mlir=args.input_mlir,
    )
    if args.input_mlir or (args.compile_to == "vmfb"):
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name,
        f"_bs{str(args.batch_size)}_{args.height}x{args.width}_{args.precision}_vae_{args.vae_variant}",
    )
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")