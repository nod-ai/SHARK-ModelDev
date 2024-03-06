# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
from turbine_models.custom_models.sd_inference.utils import create_safe_name
from turbine_models.custom_models.sdxl_inference import (
    clip,
    clip_runner,
    unet,
    unet_runner,
    vae,
    vae_runner,
)
from turbine_models.utils.sdxl_benchmark import run_benchmark
import unittest
from tqdm.auto import tqdm
from PIL import Image
import os
import numpy as np
import time


torch.random.manual_seed(0)

arguments = {}


@pytest.fixture(scope="session")
def command_line_args(request):
    arguments["hf_auth_token"] = request.config.getoption("--hf_auth_token")
    arguments["hf_model_name"] = request.config.getoption("--hf_model_name")
    arguments["scheduler_id"] = request.config.getoption("--scheduler_id")
    arguments["prompt"] = request.config.getoption("--prompt")
    arguments["negative_prompt"] = request.config.getoption("--negative_prompt")
    arguments["num_inference_steps"] = int(
        request.config.getoption("--num_inference_steps")
    )
    arguments["guidance_scale"] = float(request.config.getoption("--guidance_scale"))
    arguments["seed"] = float(request.config.getoption("--seed"))
    arguments["vmfb_path"] = request.config.getoption("--vmfb_path")
    arguments["external_weight_path"] = request.config.getoption(
        "--external_weight_path"
    )
    arguments["external_weight_dir"] = request.config.getoption("--external_weight_dir")
    arguments["external_weight_file"] = request.config.getoption(
        "--external_weight_file"
    )
    arguments["pipeline_dir"] = request.config.getoption("--pipeline_dir")
    arguments["batch_size"] = int(request.config.getoption("--batch_size"))
    arguments["height"] = int(request.config.getoption("--height"))
    arguments["width"] = int(request.config.getoption("--width"))
    arguments["precision"] = request.config.getoption("--precision")
    arguments["max_length"] = int(request.config.getoption("--max_length"))
    arguments["run_vmfb"] = request.config.getoption("--run_vmfb")
    arguments["compile_to"] = request.config.getoption("--compile_to")
    arguments["external_weights"] = request.config.getoption("--external_weights")
    arguments["decomp_attn"] = request.config.getoption("--decomp_attn")
    arguments["device"] = request.config.getoption("--device")
    arguments["rt_device"] = request.config.getoption("--rt_device")
    arguments["iree_target_triple"] = request.config.getoption("--iree_target_triple")
    arguments["ireec_flags"] = request.config.getoption("--ireec_flags")
    arguments["attn_flags"] = request.config.getoption("--attn_flags")
    arguments["in_channels"] = int(request.config.getoption("--in_channels"))
    arguments["benchmark"] = request.config.getoption("--benchmark")
    arguments["tracy_profile"] = request.config.getoption("--tracy_profile")


@pytest.mark.usefixtures("command_line_args")
class StableDiffusionXLTest(unittest.TestCase):
    def setUp(self):
        self.safe_model_name = create_safe_name(arguments["hf_model_name"], "")
        self.unet_model = unet.UnetModel(
            # This is a public model, so no auth required
            arguments["hf_model_name"],
            precision=arguments["precision"],
        )
        self.vae_model = vae.VaeModel(
            # This is a public model, so no auth required
            arguments["hf_model_name"],
            custom_vae=(
                "madebyollin/sdxl-vae-fp16-fix"
                if arguments["precision"] == "fp16"
                else None
            ),
        )

    def test01_ExportClipModels(self):
        if arguments["device"] in ["vulkan", "rocm", "cuda"]:
            self.skipTest(
                "Compilation error on vulkan; Runtime error on rocm; To be tested on cuda."
            )
        with self.assertRaises(SystemExit) as cm:
            clip.export_clip_model(
                # This is a public model, so no auth required
                hf_model_name=arguments["hf_model_name"],
                hf_auth_token=None,
                max_length=arguments["max_length"],
                precision=arguments["precision"],
                compile_to="vmfb",
                external_weights=arguments["external_weights"],
                external_weight_path=self.safe_model_name
                + "_"
                + arguments["precision"]
                + "_clip",
                device=arguments["device"],
                target_triple=arguments["iree_target_triple"],
                ireec_flags=arguments["ireec_flags"],
                index=1,
                exit_on_vmfb=True,
                pipeline_dir=arguments["pipeline_dir"],
            )
        self.assertEqual(cm.exception.code, None)
        with self.assertRaises(SystemExit) as cm:
            clip.export_clip_model(
                hf_model_name=arguments["hf_model_name"],
                hf_auth_token=None,  # This is a public model, so no auth required
                max_length=arguments["max_length"],
                precision=arguments["precision"],
                compile_to="vmfb",
                external_weights=arguments["external_weights"],
                external_weight_path=self.safe_model_name
                + "_"
                + arguments["precision"]
                + "_clip",
                device=arguments["device"],
                target_triple=arguments["iree_target_triple"],
                ireec_flags=arguments["ireec_flags"],
                index=2,
                exit_on_vmfb=True,
                pipeline_dir=arguments["pipeline_dir"],
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path_1"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_clip_1."
            + arguments["external_weights"]
        )
        arguments["external_weight_path_2"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_clip_2."
            + arguments["external_weights"]
        )
        arguments["vmfb_path_1"] = (
            self.safe_model_name
            + "_"
            + str(arguments["max_length"])
            + "_"
            + arguments["precision"]
            + "_clip_1_"
            + arguments["device"]
            + ".vmfb"
        )
        arguments["vmfb_path_2"] = (
            self.safe_model_name
            + "_"
            + str(arguments["max_length"])
            + "_"
            + arguments["precision"]
            + "_clip_2_"
            + arguments["device"]
            + ".vmfb"
        )
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
        if arguments["benchmark"] or arguments["tracy_profile"]:
            run_benchmark(
                "clip_1",
                arguments["vmfb_path_1"],
                arguments["external_weight_path_1"],
                arguments["rt_device"],
                max_length=arguments["max_length"],
                tracy_profile=arguments["tracy_profile"],
            )
            run_benchmark(
                "clip_2",
                arguments["vmfb_path_2"],
                arguments["external_weight_path_2"],
                arguments["rt_device"],
                max_length=arguments["max_length"],
                tracy_profile=arguments["tracy_profile"],
            )
        rtol = 4e-2
        atol = 4e-2
        np.testing.assert_allclose(torch_output_1, turbine_1[0], rtol, atol)
        if arguments["device"] == "cpu":
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(torch_output_2, turbine_2[0], rtol, atol)
            return
        np.testing.assert_allclose(torch_output_2, turbine_2[0], rtol, atol)

    def test02_ExportUnetModel(self):
        if arguments["device"] in ["vulkan", "rocm", "cuda"]:
            self.skipTest(
                "Unknown error on vulkan; Runtime error on rocm; To be tested on cuda."
            )
        with self.assertRaises(SystemExit) as cm:
            unet.export_unet_model(
                unet_model=self.unet_model,
                # This is a public model, so no auth required
                hf_model_name=arguments["hf_model_name"],
                batch_size=arguments["batch_size"],
                height=arguments["height"],
                width=arguments["width"],
                precision=arguments["precision"],
                max_length=arguments["max_length"],
                hf_auth_token=None,
                compile_to="vmfb",
                external_weights=arguments["external_weights"],
                external_weight_path=self.safe_model_name
                + "_"
                + arguments["precision"]
                + "_unet."
                + arguments["external_weights"],
                device=arguments["device"],
                target_triple=arguments["iree_target_triple"],
                ireec_flags=arguments["ireec_flags"],
                decomp_attn=arguments["decomp_attn"],
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_unet."
            + arguments["external_weights"]
        )
        arguments["vmfb_path"] = (
            self.safe_model_name
            + "_"
            + str(arguments["max_length"])
            + "_"
            + str(arguments["height"])
            + "x"
            + str(arguments["width"])
            + "_"
            + arguments["precision"]
            + "_unet_"
            + arguments["device"]
            + ".vmfb"
        )
        dtype = torch.float16 if arguments["precision"] == "fp16" else torch.float32
        sample = torch.rand(
            (
                arguments["batch_size"],
                arguments["in_channels"],
                arguments["height"] // 8,
                arguments["width"] // 8,
            ),
            dtype=dtype,
        )
        timestep = torch.zeros(1, dtype=torch.int64)
        prompt_embeds = torch.rand(
            (2 * arguments["batch_size"], arguments["max_length"], 2048),
            dtype=dtype,
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
            precision=arguments["precision"],
        )
        if arguments["benchmark"] or arguments["tracy_profile"]:
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
                tracy_profile=arguments["tracy_profile"],
            )
        rtol = 4e-2
        atol = 4e-2
        if arguments["device"] == "cpu" and arguments["precision"] == "fp16":
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(torch_output, turbine, rtol, atol)
            return
        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    def test03_ExportVaeModelDecode(self):
        if arguments["device"] in ["vulkan", "cuda", "rocm"]:
            self.skipTest(
                "Compilation error on vulkan; Runtime error on rocm; To be tested on cuda."
            )
        with self.assertRaises(SystemExit) as cm:
            vae.export_vae_model(
                vae_model=self.vae_model,
                # This is a public model, so no auth required
                hf_model_name=arguments["hf_model_name"],
                batch_size=arguments["batch_size"],
                height=arguments["height"],
                width=arguments["width"],
                precision=arguments["precision"],
                compile_to="vmfb",
                external_weights=arguments["external_weights"],
                external_weight_path=self.safe_model_name
                + "_"
                + arguments["precision"]
                + "_vae_decode."
                + arguments["external_weights"],
                device=arguments["device"],
                target_triple=arguments["iree_target_triple"],
                ireec_flags=arguments["ireec_flags"],
                variant="decode",
                decomp_attn=arguments["decomp_attn"],
                exit_on_vmfb=True,
                pipeline_dir=arguments["pipeline_dir"],
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_vae_decode."
            + arguments["external_weights"]
        )
        arguments["vmfb_path"] = (
            self.safe_model_name
            + "_"
            + str(arguments["height"])
            + "x"
            + str(arguments["width"])
            + "_"
            + arguments["precision"]
            + "_vae_decode_"
            + arguments["device"]
            + ".vmfb"
        )
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
            (
                "madebyollin/sdxl-vae-fp16-fix"
                if arguments["precision"] == "fp16"
                else ""
            ),
            "decode",
            example_input_torch,
        )
        if arguments["benchmark"] or arguments["tracy_profile"]:
            run_benchmark(
                "vae_decode",
                arguments["vmfb_path"],
                arguments["external_weight_path"],
                arguments["rt_device"],
                height=arguments["height"],
                width=arguments["width"],
                precision=arguments["precision"],
                tracy_profile=arguments["tracy_profile"],
            )
        rtol = 4e-2
        atol = 4e-2
        if arguments["device"] == "cpu" and arguments["precision"] == "fp16":
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(torch_output, turbine, rtol, atol)
            return
        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    def test04_ExportVaeModelEncode(self):
        if arguments["device"] in ["cpu", "vulkan", "cuda", "rocm"]:
            self.skipTest(
                "Compilation error on cpu, vulkan and rocm; To be tested on cuda."
            )
        with self.assertRaises(SystemExit) as cm:
            vae.export_vae_model(
                vae_model=self.vae_model,
                # This is a public model, so no auth required
                hf_model_name=arguments["hf_model_name"],
                batch_size=arguments["batch_size"],
                height=arguments["height"],
                width=arguments["width"],
                precision=arguments["precision"],
                compile_to="vmfb",
                external_weights=arguments["external_weights"],
                external_weight_path=self.safe_model_name
                + "_"
                + arguments["precision"]
                + "_vae_encode."
                + arguments["external_weights"],
                device=arguments["device"],
                target_triple=arguments["iree_target_triple"],
                ireec_flags=arguments["ireec_flags"],
                variant="encode",
                decomp_attn=arguments["decomp_attn"],
                exit_on_vmfb=True,
                pipeline_dir=arguments["pipeline_dir"],
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_vae_encode."
            + arguments["external_weights"]
        )
        arguments["vmfb_path"] = (
            self.safe_model_name
            + "_"
            + str(arguments["height"])
            + "x"
            + str(arguments["width"])
            + "_"
            + arguments["precision"]
            + "_vae_encode_"
            + arguments["device"]
            + ".vmfb"
        )
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
            (
                "madebyollin/sdxl-vae-fp16-fix"
                if arguments["precision"] == "fp16"
                else ""
            ),
            "encode",
            example_input_torch,
        )
        if arguments["benchmark"] or arguments["tracy_profile"]:
            run_benchmark(
                "vae_encode",
                arguments["vmfb_path"],
                arguments["external_weight_path"],
                arguments["rt_device"],
                height=arguments["height"],
                width=arguments["width"],
                precision=arguments["precision"],
                tracy_profile=arguments["tracy_profile"],
            )
        rtol = 4e-2
        atol = 4e-2
        if arguments["device"] == "cpu":
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(torch_output, turbine, rtol, atol)
            return
        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    def test05_t2i_generate_images(self):
        if arguments["device"] in ["vulkan", "rocm", "cuda"]:
            self.skipTest("Have issues with submodels on these backends")
        from diffusers import EulerDiscreteScheduler

        arguments["vae_external_weight_path"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_vae_decode."
            + arguments["external_weights"]
        )
        arguments["vae_vmfb_path"] = (
            self.safe_model_name
            + "_"
            + str(arguments["height"])
            + "x"
            + str(arguments["width"])
            + "_"
            + arguments["precision"]
            + "_vae_decode_"
            + arguments["device"]
            + ".vmfb"
        )
        arguments["unet_external_weight_path"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_unet."
            + arguments["external_weights"]
        )
        arguments["unet_vmfb_path"] = (
            self.safe_model_name
            + "_"
            + str(arguments["max_length"])
            + "_"
            + str(arguments["height"])
            + "x"
            + str(arguments["width"])
            + "_"
            + arguments["precision"]
            + "_unet_"
            + arguments["device"]
            + ".vmfb"
        )
        arguments["clip_external_weight_path"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_clip."
            + arguments["external_weights"]
        )
        arguments["clip_vmfb_path"] = (
            self.safe_model_name
            + "_"
            + str(arguments["max_length"])
            + "_"
            + arguments["precision"]
            + "_clip_"
            + arguments["device"]
            + ".vmfb"
        )

        dtype = torch.float16 if arguments["precision"] == "fp16" else torch.float32
        for key in [
            "vae_external_weight_path",
            "vae_vmfb_path",
            "unet_external_weight_path",
            "unet_vmfb_path",
            "clip_external_weight_path",
            "clip_vmfb_path",
        ]:
            try:
                assert os.path.exists(arguments[key])
            except AssertionError:
                unittest.skip(f"File {arguments[key]} not found")
        start = time.time()
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            pooled_negative_prompt_embeds,
        ) = clip_runner.run_encode_prompts(
            arguments["rt_device"],
            arguments["prompt"],
            arguments["negative_prompt"],
            arguments["clip_vmfb_path"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["clip_external_weight_path"],
            arguments["max_length"],
        )
        generator = torch.manual_seed(0)
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
        sample = init_latents * scheduler.init_noise_sigma

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
        guidance_scale = torch.tensor(arguments["guidance_scale"]).to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        add_time_ids = add_time_ids.to(dtype)
        latents = unet_runner.run_unet_steps(
            device=arguments["rt_device"],
            sample=sample,
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
            if i == 0:
                end = time.time()
                print(f"Total time taken by SD pipeline: {end-start}")
            all_imgs.append(numpy_to_pil_image(image))
        for idx, image in enumerate(all_imgs):
            img_path = "sdxl_test_image_" + str(idx) + ".png"
            image[0].save(img_path)
            print(img_path, "saved")
        with open("e2e_time.txt", "w") as f:
            f.write(f"{end-start} per batch\n")
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
