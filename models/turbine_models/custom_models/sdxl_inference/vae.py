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
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from diffusers import AutoencoderKL
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="stabilityai/stable-diffusion-xl-base-1.0",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for inference"
)
parser.add_argument(
    "--height", type=int, default=1024, help="Height of Stable Diffusion"
)
parser.add_argument("--width", type=int, default=1024, help="Width of Stable Diffusion")
parser.add_argument(
    "--precision", type=str, default="fp16", help="Precision of Stable Diffusion"
)
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument("--external_weight_path", type=str, default="")
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="saves ir/vmfb without global weights for size and readability, options [safetensors]",
)
parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, vulkan, rocm")
# TODO: Bring in detection for target triple
parser.add_argument(
    "--iree_target_triple",
    type=str,
    default="",
    help="Specify vulkan target triple or rocm/cuda target device.",
)
parser.add_argument("--vulkan_max_allocation", type=str, default="4294967296")
parser.add_argument("--variant", type=str, default="decode")


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
        inp = inp / 0.13025
        x = self.vae.decode(inp, return_dict=False)[0]
        x = (x / 2 + 0.5).clamp(0, 1)
        return x.round()

    def encode_inp(self, inp):
        latents = self.vae.encode(inp).latent_dist.sample()
        return 0.13025 * latents


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
    max_alloc=None,
    variant="decode",
):
    mapper = {}
    dtype = torch.float16 if precision == "fp16" else torch.float32
    if precision == "fp16":
        vae_model = vae_model.half()
    utils.save_external_weights(
        mapper, vae_model, external_weights, external_weight_path
    )

    sample = (batch_size, 4, height // 8, width // 8)
    if variant == "encode":
        sample = (batch_size, 3, height, width)

    class CompiledVae(CompiledModule):
        params = export_parameters(vae_model)

        def main(self, inp=AbstractTensor(*sample, dtype=dtype)):
            if variant == "decode":
                return jittable(vae_model.decode_inp)(inp)
            elif variant == "encode":
                return jittable(vae_model.encode_inp)(inp)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledVae(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = utils.create_safe_name(hf_model_name, f"-vae-{variant}")
    if compile_to != "vmfb":
        return module_str
    else:
        return utils.compile_to_vmfb(
            module_str, device, target_triple, max_alloc, safe_name
        )


if __name__ == "__main__":
    args = parser.parse_args()
    vae_model = VaeModel(
        args.hf_model_name,
    )
    mod_str = export_vae_model(
        vae_model,
        args.hf_model_name,
        args.batch_size,
        args.height,
        args.width,
        args.precision,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.vulkan_max_allocation,
        args.variant,
    )
    safe_name = utils.create_safe_name(args.hf_model_name, "-vae")
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
