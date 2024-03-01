# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
from turbine_models.custom_models.sdxl_inference import (
    clip,
    clip_runner,
    unet,
    unet_runner,
    vae,
    vae_runner,
)
from turbine_models.custom_models.sd_inference import utils
from turbine_models.utils.sdxl_benchmark import run_benchmark
import unittest
from tqdm.auto import tqdm
from PIL import Image
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token, required"
)
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
parser.add_argument(
    "--max_length", type=int, default=77, help="Sequence Length of Stable Diffusion"
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
parser.add_argument(
    "--decomp_attn",
    default=False,
    action="store_true",
    help="Decompose attention at fx graph level",
)
parser.add_argument("--num_inference_steps", type=int, default=30)
parser.add_argument("--scheduler_id", type=str, default=None)

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

def get_torch_models(hf_model_name, precision, scheduler_id, num_inference_steps):
        scheduler = utils.get_schedulers(hf_model_name)[scheduler_id]
        scheduled_unet_torch = unet.ScheduledUnetXLModel(
            # This is a public model, so no auth required
            hf_model_name,
            precision=precision,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
        )
        vae_torch = vae.VaeModel(
            # This is a public model, so no auth required
            hf_model_name,
            custom_vae=(
                "madebyollin/sdxl-vae-fp16-fix"
                if precision == "fp16"
                else None
            ),
        )
        return scheduled_unet_torch, vae_torch

def export_submodels(hf_model_name, safe_model_stem, precision, external_weights, batch_size, height, width, max_length, decomp_attn, compile_to, device, iree_target_triple, ireec_args, scheduler_id, num_inference_steps):
    scheduled_unet_torch, vae_torch = get_torch_models(hf_model_name, precision, scheduler_id, num_inference_steps)
    vae_external_weight_path = (
        safe_model_stem
        + "_"
        + precision
        + "_vae_decode."
        + external_weights
    )
    unet_external_weight_path = (
        safe_model_stem
        + "_"
        + precision
        + "_unet."
        + external_weights
    )
    clip_external_weight_path = (
        safe_model_stem
        + "_"
        + precision
        + "_clip."
        + external_weights
    )
    vae_decoder_vmfb = vae.export_vae_model(
        vae_torch,
        hf_model_name,
        batch_size,
        height,
        width,
        precision,
        compile_to,
        external_weights,
        vae_external_weight_path,
        device,
        iree_target_triple,
        None,
        "decode",
        decomp_attn,
        exit_on_vmfb=False,
    )
    clip_1_vmfb, _ = clip.export_clip_model(
        hf_model_name,
        None,
        max_length,
        precision,
        compile_to,
        external_weights,
        clip_external_weight_path,
        device,
        iree_target_triple,
        None,
        1,
        exit_on_vmfb=False,
    )
    clip_2_vmfb, _ = clip.export_clip_model(
        hf_model_name,
        None,
        max_length,
        precision,
        compile_to,
        external_weights,
        clip_external_weight_path,
        device,
        iree_target_triple,
        None,
        2,
        exit_on_vmfb=False,
    )
    unet_vmfb = unet.export_scheduled_unet_model(
        scheduled_unet_torch,
        hf_model_name,
        batch_size,
        height,
        width,
        precision,
        max_length,
        None,
        compile_to,
        external_weights,
        unet_external_weight_path,
        device,
        iree_target_triple,
        None,
        decomp_attn,
        exit_on_vmfb=False,
    )
    return vae_decoder_vmfb, clip_1_vmfb, clip_2_vmfb, unet_vmfb


def generate_images(prompt, negative_prompt, hf_model_name, safe_model_stem, precision, external_weights, batch_size, height, width, max_length, device, rt_device,  ):

    dtype = torch.float16 if precision == "fp16" else torch.float32
    
    clip_vmfb_path = (
        safe_model_stem
        + "_"
        + str(max_length)
        + "_"
        + precision
        + "_clip_"
        + device
        + ".vmfb"
    )
    unet_vmfb_path = (
        safe_model_stem
        + "_"
        + str(max_length)
        + "_"
        + str(height)
        + "x"
        + str(width)
        + "_"
        + precision
        + "_unet_"
        + device
        + ".vmfb"
    )
    vae_vmfb_path = (
        safe_model_stem
        + "_"
        + str(height)
        + "x"
        + str(width)
        + "_"
        + precision
        + "_vae_decode_"
        + device
        + ".vmfb"
    )
    vae_external_weight_path = (
        safe_model_stem
        + "_"
        + precision
        + "_vae_decode."
        + external_weights
    )
    unet_external_weight_path = (
        safe_model_stem
        + "_"
        + precision
        + "_unet."
        + external_weights
    )
    clip_external_weight_path = (
        safe_model_stem
        + "_"
        + precision
        + "_clip."
        + external_weights
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        pooled_negative_prompt_embeds,
    ) = clip_runner.run_encode_prompts(
        rt_device,
        prompt,
        negative_prompt,
        clip_vmfb_path,
        hf_model_name,
        None,
        clip_external_weight_path,
        max_length,
    )
    generator = torch.manual_seed(0)
    init_latents = torch.randn(
        (
            batch_size,
            4,
            height // 8,
            width // 8,
        ),
        generator=generator,
        dtype=dtype,
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        arguments["hf_model_name"],
        subfolder="scheduler",
    )
    scheduler.set_timesteps(arguments["num_inference_steps"])
    scheduler.is_scale_input_called = True
    latents = init_latents * scheduler.init_noise_sigma

    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    add_text_embeds = pooled_prompt_embeds

    add_time_ids = _get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
    )
    negative_add_time_ids = add_time_ids

    do_classifier_free_guidance = True
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [pooled_negative_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([add_time_ids, negative_add_time_ids], dim=0)

    add_text_embeds = add_text_embeds.to(dtype)
    add_time_ids = add_time_ids.repeat(arguments["batch_size"] * 1, 1)

    # guidance scale as a float32 tensor.
    guidance_scale = torch.tensor(arguments["guidance_scale"]).to(dtype)
    prompt_embeds = prompt_embeds.to(dtype)
    add_time_ids = add_time_ids.to(dtype)

    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )

    latents = unet_runner.run_unet_steps(
        device=arguments["rt_device"],
        sample=latent_model_input,
        scheduler=scheduler,
        prompt_embeds=prompt_embeds,
        text_embeds=add_text_embeds,
        time_ids=add_time_ids,
        guidance_scale=guidance_scale,
        vmfb_path=arguments["unet_vmfb_path"],
        external_weight_path=arguments["unet_external_weight_path"],
    )
    all_imgs = []
    for i in range(0, latents.shape[0], arguments["batch_size"]):
        vae_out = vae_runner.run_vae(
            arguments["rt_device"],
            latents[i : i + arguments["batch_size"]],
            arguments["vae_vmfb_path"],
            arguments["hf_model_name"],
            arguments["vae_external_weight_path"],
        ).to_host()
        image = torch.from_numpy(vae_out).cpu().permute(0, 2, 3, 1).float().numpy()
        all_imgs.append(numpy_to_pil_image(image))
    for idx, image in enumerate(all_imgs):
        img_path = "sdxl_test_image_" + str(idx) + ".png"
        image[0].save(img_path)
        print(img_path, "saved")
    assert os.path.exists("sdxl_test_image_0.png")


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


def _get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    # self.unet.config.addition_time_embed_dim IS 256.
    # self.text_encoder_2.config.projection_dim IS 1280.
    passed_add_embed_dim = 256 * len(add_time_ids) + 1280
    expected_add_embed_dim = 2816
    # self.unet.add_embedding.linear_1.in_features IS 2816.

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
