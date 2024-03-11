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
from transformers import CLIPTokenizer

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
        return_index=args.return_index,
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
    if not os.path.exists(args.pipeline_dir):
        os.makedirs(args.pipeline_dir)

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
            vae_decode_vmfb = vae.export_vae_model(
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
            )
            return vae_decode_vmfb, vae_external_weight_path
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
                "sdxl_sched_unet_bench_" + "f32"
                if args.precision == "fp32"
                else "sdxl_sched_unet_bench_" + "f16"
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
            return pipeline_vmfb, None


def generate_images(args, vmfbs: dict, weights: dict):
    print("Pipeline arguments: ", args)
    # TODO: implement case where this is false e.g. in SDXL Turbo
    do_classifier_free_guidance = True
    pipe_start = time.time()
    iree_dtype = "float32" if args.precision == "fp32" else "float16"
    torch_dtype = torch.float32 if args.precision == "fp32" else torch.float16

    all_imgs = []

    samples = []
    for i in range(args.batch_count):
        generator = torch.manual_seed(args.seed + i)
        rand_sample = torch.randn(
            (
                args.batch_size,
                4,
                args.height // 8,
                args.width // 8,
            ),
            generator=generator,
            dtype=torch_dtype,
        )
        samples.append(rand_sample)

    pipe_runner = vmfbRunner(
        args.rt_device,
        [vmfbs["scheduled_unet"], vmfbs["pipeline"]],
        [weights["scheduled_unet"], None],
    )
    vae_decode_runner = vmfbRunner(
        args.rt_device, vmfbs["vae_decode"], weights["vae_decode"]
    )
    clip_runner_1 = vmfbRunner(args.rt_device, vmfbs["clip_1"], weights["clip_1"])
    clip_runner_2 = vmfbRunner(args.rt_device, vmfbs["clip_2"], weights["clip_2"])
    text_encoders = [clip_runner_1, clip_runner_2]
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        args.hf_model_name,
        subfolder="tokenizer",
        token=args.hf_auth_token,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.hf_model_name,
        subfolder="tokenizer_2",
        token=args.hf_auth_token,
    )
    tokenizers = [tokenizer_1, tokenizer_2]
    prompts = [args.prompt, args.prompt]
    uncond_tokens = [args.negative_prompt, args.negative_prompt]
    prompt_embeds_list = []
    negative_prompt_embeds_list = []

    max_length = args.max_length

    encode_prompts_start = time.time()

    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, max_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_length} tokens: {removed_text}"
            )
        text_input_ids = [
            ireert.asdevicearray(text_encoder.config.device, text_input_ids)
        ]
        text_encoder_output = text_encoder.ctx.modules.compiled_clip["main"](
            *text_input_ids
        )
        prompt_embeds = torch.from_numpy(text_encoder_output[0].to_host())
        pooled_prompt_embeds = torch.from_numpy(text_encoder_output[1].to_host())

        prompt_embeds_list.append(prompt_embeds)

    for negative_prompt, tokenizer, text_encoder in zip(
        uncond_tokens, tokenizers, text_encoders
    ):
        uncond_input = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        uncond_input_ids = uncond_input.input_ids
        uncond_input_ids = [
            ireert.asdevicearray(text_encoder.config.device, uncond_input_ids)
        ]

        text_encoder_output = text_encoder.ctx.modules.compiled_clip["main"](
            *uncond_input_ids
        )
        negative_prompt_embeds = torch.from_numpy(text_encoder_output[0].to_host())
        negative_pooled_prompt_embeds = torch.from_numpy(
            text_encoder_output[1].to_host()
        )

        negative_prompt_embeds_list.append(negative_prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    do_classifier_free_guidance = True

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
    add_text_embeds = pooled_prompt_embeds

    if do_classifier_free_guidance:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, 1).view(
            1, -1
        )
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * 1, seq_len, -1)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )

    add_text_embeds = add_text_embeds.to(torch_dtype)
    prompt_embeds = prompt_embeds.to(torch_dtype)

    encode_prompts_end = time.time()

    unet_inputs = [
        ireert.asdevicearray(pipe_runner.config.device, samples[i], dtype=iree_dtype),
        ireert.asdevicearray(
            pipe_runner.config.device, prompt_embeds, dtype=iree_dtype
        ),
        ireert.asdevicearray(
            pipe_runner.config.device, add_text_embeds, dtype=iree_dtype
        ),
        ireert.asdevicearray(
            pipe_runner.config.device,
            np.asarray([args.guidance_scale]),
            dtype=iree_dtype,
        ),
    ]

    send_unet_inputs = time.time()

    numpy_images = []
    for i in range(args.batch_count):
        unet_start = time.time()

        latents = pipe_runner.ctx.modules.sdxl_compiled_pipeline[
            "produce_image_latents"
        ](
            *unet_inputs,
        )

        vae_start = time.time()
        vae_out = vae_decode_runner.ctx.modules.compiled_vae["main"](latents)

        pipe_end = time.time()

        image = (
            torch.from_numpy(vae_out.to_host())
            .cpu()
            .permute(0, 2, 3, 1)
            .float()
            .numpy()
        )

        numpy_images.append(image)
        print("Batch #", i + 1, "\n")
        print(
            "UNet time(",
            args.num_inference_steps,
            "): ",
            vae_start - unet_start,
            "sec,",
        )
        print(
            "Unet average step latency: ",
            (vae_start - unet_start) / args.num_inference_steps,
            "sec",
        )
        print("VAE time: ", pipe_end - vae_start, "sec")
        print(
            f"\nTotal time (txt2img, batch #{str(i+1)}): ",
            (send_unet_inputs - encode_prompts_start) + (pipe_end - unet_start),
            "sec\n",
        )
    end = time.time()
    print(
        "Total CLIP + Tokenizer time:", encode_prompts_end - encode_prompts_start, "sec"
    )
    print("Send UNet inputs to device:", send_unet_inputs - encode_prompts_end, "sec")
    print("Loading time: ", encode_prompts_start - pipe_start, "sec")
    print(
        f"Total inference time ({args.batch_count} batch(es)):",
        end - encode_prompts_start,
        "sec",
    )

    for image in numpy_images:
        image = numpy_to_pil_image(image)
        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_path = "sdxl_output_" + timestamp + ".png"
        image[0].save(img_path)
        print(img_path, "saved")


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


def check_prepared(args, vmfbs, weights):
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
            ready, vmfbs, weights = is_prepared(args, vmfbs, weights)
            if ready:
                print("All necessary files found. Generating images.")
                return vmfbs, weights
            else:
                print("There was an error generating the necessary files.")
                exit()
    else:
        print("All necessary files found. Generating images.")
    return vmfbs, weights


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
    vmfbs, weights = check_prepared(args, vmfbs, weights)
    generate_images(args, vmfbs, weights)
    print("Image generation complete.")
