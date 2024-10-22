# Copyrigh 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree import runtime as ireert
from iree.compiler.ir import Context
import numpy as np
from iree.turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo

import safetensors
import argparse
from turbine_models.turbine_tank import turbine_tank

SEED = 1


def export_vae(
    model,
    height,
    width,
    compile_to="torch",
    external_weight_prefix=None,
    device=None,
    target_triple=None,
    max_alloc="",
    upload_ir=False,
    dtype=torch.float32,
):
    mapper = {}
    utils.save_external_weights(mapper, model, "safetensors", external_weight_prefix)
    latent_shape = [1, 16, height // 8, width // 8]
    input_arg = torch.empty(latent_shape)
    input_arg = (input_arg.to(dtype),)
    if external_weight_prefix != None and len(external_weight_prefix) > 1:
        externalize_module_parameters(model)

    exported = export(model, args=input_arg)

    module_str = str(exported.mlir_module)
    safe_name = utils.create_safe_name(str(dtype).lstrip("torch."), "_mmdit")
    if compile_to != "vmfb":
        return module_str
    else:
        print("compiling to vmfb")
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)
        return module_str


def export_unet_dynamic(
    unet_model,
    height,
    width,
    compile_to="torch",
    external_weight_path=None,
    device=None,
    target_triple=None,
    max_alloc="",
    upload_ir=False,
    dtype=torch.float32,
):
    cond_shape = [1, 154, 4096]  # 77, 4096]
    pool_shape = [1, 2048]
    latent_shape = [1, 16, height // 8, width // 8]
    if dtype == torch.float16:
        unet_model = unet_model.half()
    mapper = {}
    utils.save_external_weights(mapper, unet_model, "safetensors", external_weight_path)

    if weights_only:
        return external_weight_path

    fxb = FxProgramsBuilder(unet_model)

    sigmas = torch.export.Dim("sigmas")
    dynamic_shapes = {"sigmas": {0: sigmas}, "latent": {}, "noise": {}}
    example_init_args = [
        torch.empty([19], dtype=dtype),
        torch.empty(latent_shape, dtype=dtype),
        torch.empty(latent_shape, dtype=dtype),
    ]
    example_sampling_args = [
        torch.empty(latent_shape, dtype=dtype),
        torch.empty(1, dtype=dtype),
        torch.empty(1, dtype=dtype),
        torch.empty(cond_shape, dtype=dtype),
        torch.empty(pool_shape, dtype=dtype),
        torch.empty(cond_shape, dtype=dtype),
        torch.empty(pool_shape, dtype=dtype),
        torch.empty(1, dtype=dtype),
    ]

    @fxb.export_program(args=(example_init_args,), dynamic_shapes=dynamic_shapes)
    def _initialize(module, inputs):
        # 1.0 is denoise currently symfloat not supported in fx_importer
        return module.init_dynamic(*inputs)

    @fxb.export_program(args=(example_sampling_args,))
    def _do_sampling(module, inputs):
        return module.do_sampling(*inputs)

    class CompiledTresleches(CompiledModule):
        initialize = _initialize
        do_sampling = _do_sampling

    # _vae_decode = vae_decode

    if external_weights:
        externalize_module_parameters(unet_model)
        save_module_parameters(external_weight_path, unet_model)

    inst = CompiledTresleches(context=Context(), import_to="IMPORT")
    module_str = str(CompiledModule.get_mlir_module(inst))
    print("exported model")

    safe_name = utils.create_safe_name(str(dtype).lstrip("torch."), "_mmdit")
    if compile_to != "vmfb":
        return module_str
    else:
        print("compiling to vmfb")
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)
        return module_str


def export_preprocessor(
    model,
    compile_to="torch",
    external_weight_path=None,
    device=None,
    target_triple=None,
    max_alloc="",
    dtype=torch.float32,
    height=512,
    width=512,
):
    external_weights = "safetensors"

    def get_noise():
        latent = torch.ones(1, 16, height // 8, width // 8, device="cpu") * 0.0609
        generator = torch.manual_seed(SEED)
        return torch.randn(
            latent.size(),
            dtype=latent.dtype,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        )

    input_args = [torch.empty([1, 77, 2], dtype=torch.int64) for x in range(6)]
    input_args += get_noise()
    if dtype == torch.float16:
        model = model.half()

    mapper = {}

    utils.save_external_weights(mapper, model, external_weights, external_weight_path)

    if external_weight_path != None and len(external_weight_path) > 1:
        print("externalizing weights")
        externalize_module_parameters(model)

    exported = export(model, args=tuple(input_args))
    print("exported model")

    # import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    # inst = CompiledTresleches(context=Context(), import_to=import_to)

    # module_str = str(CompiledModule.get_mlir_module(inst))
    module_str = str(exported.mlir_module)
    safe_name = utils.create_safe_name("sd3", "clips")
    if compile_to != "vmfb":
        return module_str
    else:
        print("compiling to vmfb")
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)
        return module_str


@torch.no_grad()
def main(args):
    import turbine_sd3
    from safetensors import safe_open

    vulkan_max_allocation = "4294967296" if args.device == "vulkan" else ""
    # st_file = "/mnt2/tresleches/models/sd3_8b_beta.safetensors"
    st_file = "/mnt2/tresleches/models/sd3_2b_512_alpha.safetensors"
    dtype = torch.float32
    if args.precision == "f16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    print(args.export)

    if args.export in ["dynamic"]:
        print("exporting dynamic")
        unet_model = turbine_sd3.SD3Inferencer(
            model=st_file, vae=turbine_sd3.VAEFile, shift=1.0, dtype=dtype
        ).eval()
        mod_str = export_unet_dynamic(
            unet_model=unet_model,
            height=args.height,
            width=args.width,
            compile_to=args.compile_to,
            external_weight_path=args.external_weight_path,
            device=args.device,
            target_triple=args.iree_target_triple,
            max_alloc=vulkan_max_allocation,
            upload_ir=False,
            dtype=dtype,
        )
        safe_name = utils.create_safe_name("hc_sd3", "-unet")
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
    export_pre = args.export in ["all", "clip"]
    print(export_pre)
    if export_pre:
        print("exporting preprocessor")
        pre = turbine_sd3.Preprocess()
        mod_str = export_preprocessor(
            model=pre,
            compile_to=args.compile_to,
            external_weight_path=args.external_weight_path,
            device=args.device,
            target_triple=args.iree_target_triple,
            max_alloc=vulkan_max_allocation,
            dtype=dtype,
            height=args.height,
            width=args.width,
        )
        safe_name = utils.create_safe_name("hc_sd3", "_preprocess")
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
    should_export_vae = args.export in ["all", "vae"]
    if should_export_vae:
        print("exporting vae")
        from turbine_impls import SDVAE

        with turbine_sd3.safe_open(
            turbine_sd3.VAEFile, framework="pt", device="cpu"
        ) as f:
            vae = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            turbine_sd3.load_into(f, vae, prefix, "cpu", dtype)
            print("Something")
        mod_str = export_vae(
            model=vae,
            height=args.height,
            width=args.width,
            compile_to=args.compile_to,
            external_weight_prefix=args.external_weight_path,
            device=args.device,
            target_triple=args.iree_target_triple,
            max_alloc=vulkan_max_allocation,
            dtype=dtype,
        )
        safe_name = utils.create_safe_name("hc_sd3", "_vae")
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")


if __name__ == "__main__":
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args

    torch._dynamo.config.capture_scalar_outputs = True
    main(args)
