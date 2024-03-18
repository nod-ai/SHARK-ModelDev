# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import torch
from turbine_models.custom_models.sdxl_inference import (
    sdxl_prompt_encoder,
    sdxl_scheduled_unet,
    vae,
)
import iree.runtime as ireert
from turbine_models.custom_models.sd_inference import utils
from turbine_models.utils.sdxl_benchmark import run_benchmark
from turbine_models.model_runner import vmfbRunner
from transformers import CLIPTokenizer

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


def export_submodel(args, submodel, input_mlir, weights_only=False):
    if not os.path.exists(args.pipeline_dir):
        os.makedirs(args.pipeline_dir)
    if input_mlir is None and submodel in ["scheduled_unet", "vae_decode"]:
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
        prompt_encoder_external_weight_path = os.path.join(
            args.external_weights_dir, "prompt_encoder." + args.external_weights
        )
    elif args.external_weights is None:
        print(
            "No external weights type specified using --external_weights, weights for imported .mlir files will not be externalized."
        )
        vae_external_weight_path = None
        unet_external_weight_path = None
        prompt_encoder_external_weight_path = None
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
        prompt_encoder_external_weight_path = os.path.join(
            args.pipeline_dir, "prompt_encoder." + args.external_weights
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
                args.ireec_flags + args.attn_flags + args.unet_flags,
                args.decomp_attn,
                exit_on_vmfb=False,
                pipeline_dir=args.pipeline_dir,
                attn_spec=args.attn_spec,
                input_mlir=mlirs["scheduled_unet"],
                weights_only=weights_only,
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
                args.ireec_flags + args.attn_flags + args.vae_flags,
                "decode",
                args.decomp_attn,
                exit_on_vmfb=False,
                pipeline_dir=args.pipeline_dir,
                attn_spec=args.attn_spec,
                input_mlir=mlirs["vae_decode"],
                weights_only=weights_only,
            )
            return vae_decode_vmfb, vae_external_weight_path
        case "prompt_encoder":
            _, prompt_encoder_vmfb = sdxl_prompt_encoder.export_prompt_encoder(
                args.hf_model_name,
                None,
                args.max_length,
                args.precision,
                "vmfb",
                args.external_weights,
                prompt_encoder_external_weight_path,
                args.device,
                args.iree_target_triple,
                args.ireec_flags + args.clip_flags,
                exit_on_vmfb=False,
                pipeline_dir=args.pipeline_dir,
                input_mlir=mlirs["prompt_encoder"],
                attn_spec=args.attn_spec,
                weights_only=weights_only,
            )
            return prompt_encoder_vmfb, prompt_encoder_external_weight_path
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
        case "full_pipeline":
            pipeline_file = (
                "sdxl_pipeline_bench_" + "f32"
                if args.precision == "fp32"
                else "sdxl_pipeline_bench_" + "f16"
            )
            pipeline_vmfb = utils.compile_to_vmfb(
                os.path.join(
                    os.path.realpath(os.path.dirname(__file__)), pipeline_file + ".mlir"
                ),
                args.device,
                args.iree_target_triple,
                args.ireec_flags,
                os.path.join(args.pipeline_dir, "full_pipeline"),
                return_path=True,
                const_expr_hoisting=False,
                mlir_source="file",
            )
            return pipeline_vmfb, None


def load_pipeline(args, vmfbs: dict, weights: dict):
    runners = {}
    if args.compiled_pipeline:
        runners["pipe"] = vmfbRunner(
            args.rt_device,
            [
                vmfbs["scheduled_unet"],
                vmfbs["prompt_encoder"],
                vmfbs["vae_decode"],
                vmfbs["full_pipeline"],
            ],
            [
                weights["scheduled_unet"],
                weights["prompt_encoder"],
                weights["vae_decode"],
                None,
            ],
        )
    else:
        runners["pipe"] = vmfbRunner(
            args.rt_device,
            [vmfbs["scheduled_unet"], vmfbs["pipeline"]],
            [weights["scheduled_unet"], None],
        )
        runners["vae_decode"] = vmfbRunner(
            args.rt_device, vmfbs["vae_decode"], weights["vae_decode"]
        )
        runners["prompt_encoder"] = vmfbRunner(
            args.rt_device, vmfbs["prompt_encoder"], weights["prompt_encoder"]
        )
    runners["tokenizer_1"] = CLIPTokenizer.from_pretrained(
        args.hf_model_name,
        subfolder="tokenizer",
        token=args.hf_auth_token,
    )
    runners["tokenizer_2"] = CLIPTokenizer.from_pretrained(
        args.hf_model_name,
        subfolder="tokenizer_2",
        token=args.hf_auth_token,
    )
    return runners


def generate_images(args, runners: dict):
    print("Pipeline arguments: ", args)

    # TODO: implement case where this is false e.g. in SDXL Turbo
    # do_classifier_free_guidance = True

    iree_dtype = "float32" if args.precision == "fp32" else "float16"
    torch_dtype = torch.float32 if args.precision == "fp32" else torch.float16

    pipe_start = time.time()

    tokenizers = [runners["tokenizer_1"], runners["tokenizer_2"]]

    max_length = args.max_length

    samples = []
    numpy_images = []

    if args.compiled_pipeline and (args.batch_count > 1):
        print(
            "Compiled one-shot pipeline only supports 1 image at a time for now. Setting batch count to 1."
        )
        args.batch_count = 1

    for i in range(args.batch_count):

        generator = torch.random.manual_seed(args.seed + i)
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
        samples.append(
            ireert.asdevicearray(
                runners["pipe"].config.device, rand_sample, dtype=iree_dtype
            )
        )

    guidance_scale = ireert.asdevicearray(
        runners["pipe"].config.device,
        np.asarray([args.guidance_scale]),
        dtype=iree_dtype,
    )

    text_input_ids_list = []
    uncond_input_ids_list = []

    tokenize_start = time.time()

    # Tokenize prompt and negative prompt.
    for tokenizer in tokenizers:
        text_inputs = tokenizer(
            args.prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input = tokenizer(
            args.negative_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        uncond_input_ids = uncond_input.input_ids

        text_input_ids_list.extend(
            [ireert.asdevicearray(runners["pipe"].config.device, text_input_ids)]
        )
        uncond_input_ids_list.extend(
            [ireert.asdevicearray(runners["pipe"].config.device, uncond_input_ids)]
        )
    if args.compiled_pipeline:
        inf_start = time.time()
        image = runners["pipe"].ctx.modules.sdxl_compiled_pipeline["tokens_to_image"](
            samples[0], guidance_scale, *text_input_ids_list, *uncond_input_ids_list
        )
        inf_end = time.time()
        print(
            "Total inference time (Tokens to Image): "
            + str(inf_end - inf_start)
            + "sec"
        )
        numpy_images.append(image.to_host())
    else:
        encode_prompts_start = time.time()

        prompt_embeds, add_text_embeds = runners[
            "prompt_encoder"
        ].ctx.modules.compiled_clip["encode_prompts"](
            *text_input_ids_list, *uncond_input_ids_list
        )

        encode_prompts_end = time.time()

        for i in range(args.batch_count):
            unet_start = time.time()

            latents = runners["pipe"].ctx.modules.sdxl_compiled_pipeline[
                "produce_image_latents"
            ](samples[i], prompt_embeds, add_text_embeds, guidance_scale)

            vae_start = time.time()
            vae_out = runners["vae_decode"].ctx.modules.compiled_vae["main"](latents)

            pipe_end = time.time()

            image = vae_out.to_host()

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
                (encode_prompts_end - encode_prompts_start) + (pipe_end - unet_start),
                "sec\n",
            )
        end = time.time()
        print("Total CLIP time:", encode_prompts_end - encode_prompts_start, "sec")
        print("Total tokenize time:", encode_prompts_start - tokenize_start, "sec")
        print("Loading time: ", encode_prompts_start - pipe_start, "sec")
        if args.batch_count > 1:
            print(
                f"Total inference time ({args.batch_count} batch(es)):",
                end - encode_prompts_start,
                "sec",
            )
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    for idx, image in enumerate(numpy_images):
        image = torch.from_numpy(image).cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil_image(image)
        img_path = "sdxl_output_" + timestamp + "_" + str(idx) + ".png"
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
        if "pipeline" in w_key:
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


def check_prepared(args, mlirs, vmfbs, weights):
    ready, vmfbs, weights = is_prepared(args, vmfbs, weights)
    if not ready:
        do_continue = input(
            f"\nIt seems you are missing some necessary files. Would you like to generate them now? (y/n)"
        )
        if do_continue.lower() != "y":
            exit()
        elif do_continue.lower() == "y":
            for submodel in vmfbs.keys():
                mlir_path = os.path.join(args.pipeline_dir, submodel + ".mlir")
                if vmfbs[submodel] == None:
                    vmfb, weight = export_submodel(
                        args, submodel, input_mlir=mlirs[submodel]
                    )
                    vmfbs[submodel] = vmfb
                    if weights[submodel] is None:
                        weights[submodel] = weight
                elif weights[submodel] is None and "pipeline" not in submodel:
                    _, weight = export_submodel(args, submodel, weights_only=True)
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


def get_mlir_from_turbine_tank(args, submodel, container_name):
    from turbine_models.turbine_tank import downloadModelArtifacts

    safe_name = utils.create_safe_name(
        args.hf_model_name,
        f"_{args.max_length}_{args.height}x{args.width}_{args.precision}_{submodel}.mlir",
    )
    mlir_path = downloadModelArtifacts(
        safe_name,
        container_name,
    )
    return mlir_path


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    mlirs = {
        "vae_decode": None,
        "prompt_encoder": None,
        "scheduled_unet": None,
        "pipeline": None,
        "full_pipeline": None,
    }
    vmfbs = {
        "vae_decode": None,
        "prompt_encoder": None,
        "scheduled_unet": None,
        "pipeline": None,
        "full_pipeline": None,
    }
    weights = {
        "vae_decode": None,
        "prompt_encoder": None,
        "scheduled_unet": None,
        "pipeline": None,
        "full_pipeline": None,
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
    if args.input_mlir:
        user_mlir_list = args.input_mlir.split(",")
    else:
        user_mlir_list = []
    for submodel_id, mlir_path in zip(mlirs.keys(), user_mlir_list):
        if submodel_id in mlir_path:
            mlirs[submodel_id] = mlir_path
        elif args.download_mlir:
            if args.container_name not in [None, ""]:
                container_name = args.container_name
            else:
                container_name = os.environ.get("AZURE_CONTAINER_NAME")
            mlirs[submodel_id] = get_mlir_from_turbine_tank(
                args, submodel_id, container_name
            )

    if not args.external_weights_dir and args.external_weights:
        args.external_weights_dir = args.pipeline_dir
    vmfbs, weights = check_prepared(args, mlirs, vmfbs, weights)

    runners = load_pipeline(args, vmfbs, weights)
    generate_images(args, runners)
    print("Image generation complete.")
