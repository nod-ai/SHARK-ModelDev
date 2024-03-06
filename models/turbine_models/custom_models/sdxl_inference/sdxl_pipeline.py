# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import torch
from turbine_models.custom_models.sdxl_inference import (
    clip,
    clip_runner,
    sdxl_scheduled_unet,
    unet_runner,
    vae,
    vae_runner,
)
import iree.runtime as ireert
from turbine_models.custom_models.sd_inference import utils
from turbine_models.utils.sdxl_benchmark import run_benchmark
from turbine_models.model_runner import vmfbRunner
import unittest
from PIL import Image
import os
import numpy as np
import time
from datetime import datetime as dt

device_list = [
    "cpu",
    "vulkan",
    "cuda",
    "rocm",
]

rt_device_list = [
    "local-task",
    "local-sync",
    "vulkan",
    "cuda",
    "rocm",
]


def get_torch_models(args):
    scheduled_unet_torch = sdxl_scheduled_unet.SDXLScheduledUnet(
        # This is a public model, so no auth required
        args.hf_model_name,
        args.scheduler_id,
        args.height,
        args.width,
        args.batch_size,
        None,
        precision=args.precision,
        num_inference_steps=args.num_inference_steps,
    )
    vae_torch = vae.VaeModel(
        # This is a public model, so no auth required
        args.hf_model_name,
        custom_vae=(
            "madebyollin/sdxl-vae-fp16-fix" if args.precision == "fp16" else None
        ),
    )
    return scheduled_unet_torch, vae_torch


def export_submodel(args, submodel):
    scheduled_unet_torch, vae_torch = get_torch_models(args)
    if args.external_weights_dir:
        if not os.path.exists(args.external_weights_dir):
            os.makedirs(args.external_weights_dir, exist_ok=True)
        vae_external_weight_path = os.path.join(
            args.external_weights_dir, "vae_decode." + args.external_weights
        )
        unet_external_weight_path = os.path.join(
            args.external_weights_dir, "scheduled_unet." + args.external_weights
        )
        clip_external_weight_path = os.path.join(
            args.external_weights_dir, "clip." + args.external_weights
        )
    elif args.external_weights is None:
        print(
            "No external weights type specified using --external_weights, weights for imported .mlir files will not be externalized."
        )
        vae_external_weight_path = None
        unet_external_weight_path = None
        clip_external_weight_path = None
    else:
        print(
            f"No external weights directory specified using --external_weights_dir, we assume you have your own weights in {args.pipeline_dir}."
        )
        args.external_weights_dir = args.pipeline_dir
        if not os.path.exists(args.pipeline_dir):
            os.makedirs(args.pipeline_dir, exist_ok=True)
        vae_external_weight_path = os.path.join(
            args.pipeline_dir, "vae_decode." + args.external_weights
        )
        unet_external_weight_path = os.path.join(
            args.pipeline_dir, "scheduled_unet." + args.external_weights
        )
        clip_external_weight_path = os.path.join(
            args.pipeline_dir, "clip." + args.external_weights
        )
    match submodel:
        case "scheduled_unet":
            unet_vmfb = sdxl_scheduled_unet.export_scheduled_unet_model(
                scheduled_unet_torch,
                args.scheduler_id,
                args.num_inference_steps,
                args.hf_model_name,
                args.batch_size,
                args.height,
                args.width,
                args.precision,
                args.max_length,
                None,
                "vmfb",
                args.external_weights,
                unet_external_weight_path,
                args.device,
                args.iree_target_triple,
                args.ireec_flags + args.attn_flags,
                args.decomp_attn,
                exit_on_vmfb=False,
                pipeline_dir=args.pipeline_dir,
            )
            return unet_vmfb, unet_external_weight_path
        case "vae_decode":
            return (
                vae.export_vae_model(
                    vae_torch,
                    args.hf_model_name,
                    args.batch_size,
                    args.height,
                    args.width,
                    args.precision,
                    "vmfb",
                    args.external_weights,
                    vae_external_weight_path,
                    args.device,
                    args.iree_target_triple,
                    args.ireec_flags + args.attn_flags,
                    "decode",
                    args.decomp_attn,
                    exit_on_vmfb=False,
                    pipeline_dir=args.pipeline_dir,
                ),
                vae_external_weight_path,
            )
        case "clip_1":
            clip_1_vmfb, _ = clip.export_clip_model(
                args.hf_model_name,
                None,
                args.max_length,
                args.precision,
                "vmfb",
                args.external_weights,
                clip_external_weight_path,
                args.device,
                args.iree_target_triple,
                args.ireec_flags,
                index=1,
                exit_on_vmfb=False,
                pipeline_dir=args.pipeline_dir,
            )
            return clip_1_vmfb, clip_external_weight_path
        case "clip_2":
            clip_2_vmfb, _ = clip.export_clip_model(
                args.hf_model_name,
                None,
                args.max_length,
                args.precision,
                "vmfb",
                args.external_weights,
                clip_external_weight_path,
                args.device,
                args.iree_target_triple,
                args.ireec_flags,
                2,
                exit_on_vmfb=False,
                pipeline_dir=args.pipeline_dir,
            )
            return clip_2_vmfb, clip_external_weight_path
        case "pipeline":
            pipeline_file = (
                "sdxl_sched_unet_bench_" + "f32" if args.precision == "fp32" else "f16"
            )
            pipeline_vmfb = utils.compile_to_vmfb(
                os.path.join(
                    os.path.realpath(os.path.dirname(__file__)), pipeline_file + ".mlir"
                ),
                args.device,
                args.iree_target_triple,
                args.ireec_flags,
                os.path.join(args.pipeline_dir, "pipeline"),
                return_path=True,
                const_expr_hoisting=False,
                mlir_source="file",
            )
            breakpoint()
            return pipeline_vmfb, None


def generate_images(args, vmfbs: dict, weights: dict):
    pipe_start = time.time()
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    all_imgs = []
    generator = torch.manual_seed(0)
    rand_sample = torch.randn(
        (
            args.batch_size,
            4,
            args.height // 8,
            args.width // 8,
        ),
        generator=generator,
        dtype=dtype,
    )

    pipe_runner = vmfbRunner(
        args.rt_device,
        [vmfbs["scheduled_unet"], vmfbs["pipeline"]],
        [weights["scheduled_unet"], None],
    )
    vae_decode_runner = vmfbRunner(
        args.rt_device, vmfbs["vae_decode"], weights["vae_decode"]
    )
    clip_start = time.time()
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        pooled_negative_prompt_embeds,
    ) = clip_runner.run_encode_prompts(
        args.rt_device,
        args.prompt,
        args.negative_prompt,
        vmfbs["clip_1"],
        vmfbs["clip_2"],
        args.hf_model_name,
        None,
        weights["clip_1"],
        weights["clip_2"],
        args.max_length,
    )

    add_text_embeds = pooled_prompt_embeds
    # Assumes that we're doing the equivalent of diffusers 'do_classifier_free_guidance' here
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    add_text_embeds = torch.cat([pooled_negative_prompt_embeds, add_text_embeds], dim=0)

    add_text_embeds = add_text_embeds.to(dtype)
    prompt_embeds = prompt_embeds.to(dtype)

    unet_start = time.time()

    unet_inputs = [
        ireert.asdevicearray(pipe_runner.config.device, rand_sample),
        ireert.asdevicearray(pipe_runner.config.device, prompt_embeds),
        ireert.asdevicearray(pipe_runner.config.device, add_text_embeds),
        ireert.asdevicearray(
            pipe_runner.config.device,
            np.asarray([args.guidance_scale]),
            dtype="float32" if args.precision == "fp32" else "float16",
        ),
    ]
    latents = pipe_runner.ctx.modules.sdxl_compiled_pipeline["produce_image_latents"](
        *unet_inputs,
    )

    vae_start = time.time()
    vae_out = vae_decode_runner.ctx.modules.compiled_vae["main"](latents)

    pipe_end = time.time()

    image = (
        torch.from_numpy(vae_out.to_host()).cpu().permute(0, 2, 3, 1).float().numpy()
    )

    image = numpy_to_pil_image(image)
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_path = "sdxl_output_" + timestamp + ".png"
    image[0].save(img_path)
    print(img_path, "saved")
    print("Pipeline arguments: ", args)
    print("Total time: ", pipe_end - pipe_start, "sec")
    print("Loading time: ", clip_start - pipe_start, "sec")
    print("Clip time: ", unet_start - clip_start, "sec")
    print("UNet time(", args.num_inference_steps, "): ", vae_start - unet_start, "sec,")
    print(
        "Unet average time: ",
        (vae_start - unet_start) / args.num_inference_steps,
        "sec",
    )
    print("VAE time: ", pipe_end - vae_start, "sec")
    assert os.path.exists(img_path)


def numpy_to_pil_image(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def is_prepared(args, vmfbs, weights):
    missing = []
    for key in vmfbs:
        if key == "scheduled_unet":
            val = f"{args.scheduler_id}_unet_{args.num_inference_steps}"
            default_filepath = os.path.join(args.pipeline_dir, val + ".vmfb")
        else:
            val = vmfbs[key]
            default_filepath = os.path.join(args.pipeline_dir, key + ".vmfb")
        if vmfbs[key] is not None and os.path.exists(vmfbs[key]):
            continue
        elif vmfbs[key] == None and os.path.exists(default_filepath):
            vmfbs[key] = default_filepath
        elif val is None:
            missing.append(key + ".vmfb")
        else:
            missing.append(val + ".vmfb")
    for w_key in weights:
        if w_key == "pipeline":
            continue
        if weights[w_key] is not None and os.path.exists(weights[w_key]):
            continue
        default_name = os.path.join(
            args.external_weights_dir, w_key + "." + args.external_weights
        )
        if weights[w_key] is None and os.path.exists(default_name):
            weights[w_key] = os.path.join(default_name)
        else:
            missing.append(w_key + "." + args.external_weights)
    if len(missing) > 0:
        print(f"Missing files: " + ", ".join(missing))
        return False, vmfbs, weights
    else:
        return True, vmfbs, weights


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    vmfbs = {
        "vae_decode": None,
        "clip_1": None,
        "clip_2": None,
        "scheduled_unet": None,
        "pipeline": None,
    }
    weights = {
        "vae_decode": None,
        "clip_1": None,
        "clip_2": None,
        "scheduled_unet": None,
        "pipeline": None,
    }
    if not args.pipeline_dir:
        pipe_id_list = [
            "sdxl_1_0",
            str(args.height),
            str(args.width),
            str(args.max_length),
            args.precision,
            args.device,
        ]
        args.pipeline_dir = os.path.join(
            ".",
            "_".join(pipe_id_list),
        )
    if not args.external_weights_dir and args.external_weights:
        args.external_weights_dir = args.pipeline_dir
    ready, vmfbs, weights = is_prepared(args, vmfbs, weights)
    if not ready:
        do_continue = input(
            f"\nIt seems you are missing some necessary files. Would you like to generate them now? (y/n)"
        )
        if do_continue.lower() != "y":
            exit()
        elif do_continue == "y":
            for submodel in vmfbs.keys():
                if vmfbs[submodel] == None:
                    vmfb, weight = export_submodel(args, submodel)
                    vmfbs[submodel] = vmfb
                    if weights[submodel] is None:
                        weights[submodel] = weight
            assert is_prepared(args, vmfbs, weights)[0]
    generate_images(args, vmfbs, weights)
    print("Image generation complete.")
