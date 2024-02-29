# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import sys
import torch
from transformers import CLIPTextModel
from turbine_models.custom_models.sdxl_inference import (
    clip,
    clip_runner,
    unet,
    unet_runner,
    vae,
    vae_runner,
)
from turbine_models.custom_models.sd_inference import utils
from turbine_models.tests.sdxl_benchmark import run_benchmark
import unittest
from tqdm.auto import tqdm
import time
from PIL import Image
import os
import numpy as np

torch.random.manual_seed(0)

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

arguments = {
    "hf_auth_token": None,
    "hf_model_name": "stabilityai/stable-diffusion-xl-base-1.0",
    "safe_model_name": "stable_diffusion_xl_base_1_0",
    "batch_size": 1,
    "height": 1024,
    "width": 1024,
    "precision": "fp16",
    "max_length": 64,
    "guidance_scale": 7.5,
    "run_vmfb": True,
    "compile_to": None,
    "external_weight_path": "",
    "vmfb_path": "",
    "external_weights": "safetensors",
    "device": "cpu",
    "rt_device": "local-task",
    "iree_target_triple": "x86_64-linux-gnu",
    "vulkan_max_allocation": "4294967296",
    "prompt": "a photograph of an astronaut riding a horse",
    "negative_prompt": "blurry, unsaturated, watermark, noisy, grainy, out of focus",
    "in_channels": 4,
    "num_inference_steps": 35,
    "benchmark": False,
    "decomp_attn": False,
}


unet_model = unet.UnetModel(
    # This is a public model, so no auth required
    arguments["hf_model_name"],
    precision=arguments["precision"],
)

vae_model = vae.VaeModel(
    # This is a public model, so no auth required
    arguments["hf_model_name"],
    custom_vae="madebyollin/sdxl-vae-fp16-fix"
    if arguments["precision"] == "fp16"
    else None,
)


class StableDiffusionXLTest(unittest.TestCase):
    @unittest.skipIf(
        arguments["device"] in ["vulkan", "cuda", "rocm"],
        reason="Fail to compile on vulkan and rocm; To be tested on cuda.",
    )
    def test01_ExportClipModels(self):
        with self.assertRaises(SystemExit) as cm:
            clip.export_clip_model(
                # This is a public model, so no auth required
                arguments["hf_model_name"],
                None,
                arguments["max_length"],
                arguments["precision"],
                "vmfb",
                "safetensors",
                f"{arguments['safe_model_name']}_{arguments['precision']}_clip",
                arguments["device"],
                arguments["iree_target_triple"],
                index=1,
            )
        self.assertEqual(cm.exception.code, None)
        with self.assertRaises(SystemExit) as cm:
            clip.export_clip_model(
                arguments["hf_model_name"],
                None,  # This is a public model, so no auth required
                arguments["max_length"],
                arguments["precision"],
                "vmfb",
                "safetensors",
                f"{arguments['safe_model_name']}_{arguments['precision']}_clip",
                arguments["device"],
                arguments["iree_target_triple"],
                index=2,
            )
        self.assertEqual(cm.exception.code, None)
        arguments[
            "external_weight_path_1"
        ] = f"{arguments['safe_model_name']}_{arguments['precision']}_clip_1.safetensors"
        arguments[
            "external_weight_path_2"
        ] = f"{arguments['safe_model_name']}_{arguments['precision']}_clip_2.safetensors"
        arguments[
            "vmfb_path_1"
        ] = f"{arguments['safe_model_name']}_{str(arguments['max_length'])}_{arguments['precision']}_clip_1_{arguments['device']}.vmfb"
        arguments[
            "vmfb_path_2"
        ] = f"{arguments['safe_model_name']}_{str(arguments['max_length'])}_{arguments['precision']}_clip_2_{arguments['device']}.vmfb"
        turbine_1 = clip_runner.run_clip(
            arguments["rt_device"],
            arguments["prompt"],
            arguments["vmfb_path_1"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["external_weight_path_1"],
            arguments["max_length"],
            index=1,
        )
        turbine_2 = clip_runner.run_clip(
            arguments["rt_device"],
            arguments["prompt"],
            arguments["vmfb_path_2"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["external_weight_path_2"],
            arguments["max_length"],
            index=2,
        )
        torch_output_1, torch_output_2 = clip_runner.run_torch_clip(
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["prompt"],
            arguments["max_length"],
        )
        if arguments["benchmark"]:
            run_benchmark(
                "clip_1",
                arguments["vmfb_path_1"],
                arguments["external_weight_path_1"],
                arguments["rt_device"],
                max_length=arguments["max_length"],
            )
            run_benchmark(
                "clip_2",
                arguments["vmfb_path_2"],
                arguments["external_weight_path_2"],
                arguments["rt_device"],
                max_length=arguments["max_length"],
            )
        rtol = 4e-2
        atol = 4e-2
        np.testing.assert_allclose(torch_output_1, turbine_1[0], rtol, atol)
        np.testing.assert_allclose(torch_output_2, turbine_2[0], rtol, atol)

    @unittest.skipIf(
        arguments["device"] in ["vulkan", "cuda", "rocm"],
        reason="Numerics issue on cpu; Fail to compile on vulkan; Runtime issue on rocm; To be tested on cuda.",
    )
    def test02_ExportUnetModel(self):
        with self.assertRaises(SystemExit) as cm:
            unet.export_unet_model(
                unet_model,
                # This is a public model, so no auth required
                arguments["hf_model_name"],
                arguments["batch_size"],
                arguments["height"],
                arguments["width"],
                arguments["precision"],
                arguments["max_length"],
                hf_auth_token=None,
                compile_to="vmfb",
                external_weights="safetensors",
                external_weight_path=f"{arguments['safe_model_name']}_{arguments['precision']}_unet.safetensors",
                device=arguments["device"],
                target_triple=arguments["iree_target_triple"],
                decomp_attn=arguments["decomp_attn"],
            )
        self.assertEqual(cm.exception.code, None)
        arguments[
            "external_weight_path"
        ] = f"{arguments['safe_model_name']}_unet.safetensors"
        arguments[
            "vmfb_path"
        ] = f"{arguments['safe_model_name']}_{str(arguments['max_length'])}_{arguments['height']}x{arguments['width']}_{arguments['precision']}_unet_{arguments['device']}.vmfb"
        dtype = torch.float16 if arguments["precision"] == "fp16" else torch.float32
        sample = torch.rand(
            (
                2 * arguments["batch_size"],
                arguments["in_channels"],
                arguments["height"] // 8,
                arguments["width"] // 8,
            ),
            dtype=dtype,
        )
        timestep = torch.zeros(1, dtype=torch.int64)
        prompt_embeds = torch.rand(
            2 * arguments["batch_size"], arguments["max_length"], 2048, dtype=dtype
        )
        text_embeds = torch.rand(2 * arguments["batch_size"], 1280, dtype=dtype)
        time_ids = torch.zeros(2 * arguments["batch_size"], 6, dtype=dtype)
        guidance_scale = torch.Tensor([arguments["guidance_scale"]]).to(dtype)

        turbine = unet_runner.run_unet(
            arguments["rt_device"],
            sample,
            timestep,
            prompt_embeds,
            text_embeds,
            time_ids,
            guidance_scale,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["external_weight_path"],
        )
        torch_output = unet_runner.run_torch_unet(
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            sample.float(),
            timestep,
            prompt_embeds.float(),
            text_embeds.float(),
            time_ids.float(),
            guidance_scale.float(),
        )
        if arguments["benchmark"]:
            run_benchmark(
                "unet",
                arguments["vmfb_path"],
                arguments["external_weight_path"],
                arguments["rt_device"],
                max_length=arguments["max_length"],
                height=arguments["height"],
                width=arguments["width"],
                batch_size=arguments["batch_size"],
                in_channels=arguments["in_channels"],
                precision=arguments["precision"],
            )
        rtol = 4e-2
        atol = 4e-2
        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    @unittest.skipIf(
        arguments["device"] in ["vulkan", "cuda", "rocm"],
        reason="Numerics issue on cpu; Fail to compile on vulkan and rocm; To be tested on cuda.",
    )
    def test03_ExportVaeModelDecode(self):
        with self.assertRaises(SystemExit) as cm:
            vae.export_vae_model(
                vae_model,
                # This is a public model, so no auth required
                arguments["hf_model_name"],
                arguments["batch_size"],
                arguments["height"],
                arguments["width"],
                arguments["precision"],
                compile_to="vmfb",
                external_weights="safetensors",
                external_weight_path=f"{arguments['safe_model_name']}_{arguments['precision']}_vae_decode.safetensors",
                device=arguments["device"],
                target_triple=arguments["iree_target_triple"],
                variant="decode",
                decomp_attn=arguments["decomp_attn"],
            )
        self.assertEqual(cm.exception.code, None)
        arguments[
            "external_weight_path"
        ] = f"{arguments['safe_model_name']}_{arguments['precision']}_vae_decode.safetensors"
        arguments[
            "vmfb_path"
        ] = f"{arguments['safe_model_name']}_{arguments['height']}x{arguments['width']}_{arguments['precision']}_vae_decode_{arguments['device']}.vmfb"
        example_input = torch.ones(
            arguments["batch_size"],
            4,
            arguments["height"] // 8,
            arguments["width"] // 8,
            dtype=torch.float32,
        )
        example_input_torch = example_input
        if arguments["precision"] == "fp16":
            example_input = example_input.half()
        turbine = vae_runner.run_vae(
            arguments["rt_device"],
            example_input,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["external_weight_path"],
        )
        torch_output = vae_runner.run_torch_vae(
            arguments["hf_model_name"],
            "madebyollin/sdxl-vae-fp16-fix" if arguments["precision"] == "fp16" else "",
            "decode",
            example_input_torch,
        )
        if arguments["benchmark"]:
            run_benchmark(
                "vae_decode",
                arguments["vmfb_path"],
                arguments["external_weight_path"],
                arguments["rt_device"],
                height=arguments["height"],
                width=arguments["width"],
                precision=arguments["precision"],
            )
        rtol = 4e-2
        atol = 4e-2
        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    @unittest.skipIf(
        arguments["device"] in ["cpu", "vulkan", "cuda", "rocm"],
        reason="Numerics issue on cpu; Fail to compile on vulkan and rocm; To be tested on cuda.",
    )
    def test04_ExportVaeModelEncode(self):
        with self.assertRaises(SystemExit) as cm:
            vae.export_vae_model(
                vae_model,
                # This is a public model, so no auth required
                arguments["hf_model_name"],
                arguments["batch_size"],
                arguments["height"],
                arguments["width"],
                arguments["precision"],
                compile_to="vmfb",
                external_weights="safetensors",
                external_weight_path=f"{arguments['safe_model_name']}_{arguments['precision']}_vae_encode.safetensors",
                device=arguments["device"],
                target_triple=arguments["iree_target_triple"],
                variant="encode",
                decomp_attn=arguments["decomp_attn"],
            )
        self.assertEqual(cm.exception.code, None)
        arguments[
            "external_weight_path"
        ] = f"{arguments['safe_model_name']}_{arguments['precision']}_vae_encode.safetensors"
        arguments[
            "vmfb_path"
        ] = f"{arguments['safe_model_name']}_{arguments['height']}x{arguments['width']}_{arguments['precision']}_vae_encode_{arguments['device']}.vmfb"
        example_input = torch.ones(
            arguments["batch_size"],
            3,
            arguments["height"],
            arguments["width"],
            dtype=torch.float32,
        )
        example_input_torch = example_input
        if arguments["precision"] == "fp16":
            example_input = example_input.half()
        turbine = vae_runner.run_vae(
            arguments["rt_device"],
            example_input,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["external_weight_path"],
        )
        torch_output = vae_runner.run_torch_vae(
            arguments["hf_model_name"],
            "madebyollin/sdxl-vae-fp16-fix" if arguments["precision"] == "fp16" else "",
            "encode",
            example_input_torch,
        )
        if arguments["benchmark"]:
            run_benchmark(
                "vae_encode",
                arguments["vmfb_path"],
                arguments["external_weight_path"],
                arguments["rt_device"],
                height=arguments["height"],
                width=arguments["width"],
                precision=arguments["precision"],
            )
        rtol = 4e-2
        atol = 4e-2
        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    def test05_t2i_generate_images(self):
        from diffusers import EulerDiscreteScheduler

        arguments[
            "vae_external_weight_path"
        ] = f"{arguments['safe_model_name']}_{arguments['precision']}_vae_decode.safetensors"
        arguments[
            "vae_vmfb_path"
        ] = f"{arguments['safe_model_name']}_{arguments['height']}x{arguments['width']}_{arguments['precision']}_vae_decode_{arguments['device']}.vmfb"
        arguments[
            "unet_external_weight_path"
        ] = f"{arguments['safe_model_name']}_{arguments['precision']}_unet.safetensors"
        arguments[
            "unet_vmfb_path"
        ] = f"{arguments['safe_model_name']}_{str(arguments['max_length'])}_{arguments['height']}x{arguments['width']}_{arguments['precision']}_unet_{arguments['device']}.vmfb"
        arguments[
            "clip_external_weight_path"
        ] = f"{arguments['safe_model_name']}_{arguments['precision']}_clip.safetensors"
        arguments[
            "clip_vmfb_path"
        ] = f"{arguments['safe_model_name']}_{str(arguments['max_length'])}_{arguments['precision']}_clip_{arguments['device']}.vmfb"

        dtype = torch.float16 if arguments["precision"] == "fp16" else torch.float32

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            pooled_negative_prompt_embeds,
        ) = clip_runner.run_clip(
            arguments["rt_device"],
            arguments["prompt"],
            arguments["negative_prompt"],
            arguments["clip_vmfb_path"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["clip_external_weight_path"],
            arguments["max_length"],
        )
        print(
            prompt_embeds.shape,
            pooled_prompt_embeds.shape,
            negative_prompt_embeds.shape,
            pooled_negative_prompt_embeds.shape,
        )
        seed = 1234567
        generator = torch.manual_seed(seed)
        init_latents = torch.randn(
            (
                arguments["batch_size"],
                4,
                arguments["height"] // 8,
                arguments["width"] // 8,
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

        original_size = (arguments["height"], arguments["width"])
        target_size = (arguments["height"], arguments["width"])
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
        guidance_scale = torch.tensor(7.5).to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        add_time_ids = add_time_ids.to(dtype)

        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )

        unet_out = unet_runner.run_unet_steps(
            device=arguments["rt_device"],
            sample=latent_model_input,
            scheduler=scheduler,
            num_inference_steps=arguments["num_inference_steps"],
            prompt_embeds=prompt_embeds,
            text_embeds=add_text_embeds,
            time_ids=add_time_ids,
            guidance_scale=guidance_scale,
            vmfb_path=arguments["unet_vmfb_path"],
            external_weight_path=arguments["unet_external_weight_path"],
        )
        vae_out = vae_runner.run_vae(
            arguments["rt_device"],
            unet_out,
            arguments["vae_vmfb_path"],
            arguments["hf_model_name"],
            arguments["vae_external_weight_path"],
        ).to_host()
        image = torch.from_numpy(vae_out).cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(image[:, :, :3])
        pil_image.save("sdxl_image.png")
        assert os.path.exists("sdxl_image.png")


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


def parse_args(args):
    consume_args = []
    for idx, arg in enumerate(args):
        if arg in arguments.keys():
            try:
                arguments[arg] = int(args[idx + 1])
            except:
                if args[idx + 1].lower() in ["true", "false"]:
                    arguments[arg] = bool(args[idx + 1])
                arguments[arg] = args[idx + 1]
            consume_args.extend([idx + 1, idx + 2])
    return consume_args


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    consume_args = parse_args(sys.argv[1:])[::-1]
    print("Test Config:", arguments)
    assert arguments["device"] in device_list
    assert arguments["rt_device"] in rt_device_list
    for idx in consume_args:
        del sys.argv[idx]
    unittest.main()
