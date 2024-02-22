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
import unittest


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
    "in_channels": 4,
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
                f"{arguments['safe_model_name']}" + "_clip",
                arguments["device"],
                arguments["iree_target_triple"],
                index=1,
            )
        self.assertEqual(cm.exception.code, None)
        with self.assertRaises(SystemExit) as cm:
            clip.export_clip_model(
                # This is a public model, so no auth required
                arguments["hf_model_name"],
                None,
                arguments["max_length"],
                arguments["precision"],
                "vmfb",
                "safetensors",
                f"{arguments['safe_model_name']}" + "_clip",
                arguments["device"],
                arguments["iree_target_triple"],
                index=2,
            )
        self.assertEqual(cm.exception.code, None)
        arguments[
            "external_weight_path_1"
        ] = f"{arguments['safe_model_name']}_clip_1.safetensors"
        arguments[
            "external_weight_path_2"
        ] = f"{arguments['safe_model_name']}_clip_2.safetensors"
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
            benchmark=True,
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
            benchmark=True,
        )
        torch_output_1, torch_output_2 = clip_runner.run_torch_clip(
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["prompt"],
            arguments["max_length"],
        )
        err1 = utils.largest_error(torch_output_1, turbine_1[0])
        err2 = utils.largest_error(torch_output_2, turbine_2[0])
        assert err1 < 4e-2 and err2 < 4e-2

    @unittest.expectedFailure
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
                external_weight_path=f"{arguments['safe_model_name']}_unet.safetensors",
                device=arguments["device"],
                target_triple=arguments["iree_target_triple"],
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
                arguments["batch_size"],
                arguments["in_channels"],
                arguments["height"] // 8,
                arguments["width"] // 8,
            ),
            dtype=dtype,
        )
        timestep = torch.zeros((1), dtype=dtype)
        prompt_embeds = torch.rand(
            (2 * arguments["batch_size"], arguments["max_length"], 2048), dtype=dtype
        )
        text_embeds = torch.rand((2 * arguments["batch_size"], 1280), dtype=dtype)
        time_ids = torch.zeros((2 * arguments["batch_size"], 6), dtype=dtype)
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
            benchmark=True,
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
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-5

    @unittest.expectedFailure
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
            benchmark=True,
        )
        torch_output = vae_runner.run_torch_vae(
            arguments["hf_model_name"],
            "madebyollin/sdxl-vae-fp16-fix" if arguments["precision"] == "fp16" else "",
            "decode",
            example_input_torch,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-3

    @unittest.expectedFailure
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
            benchmark=True,
        )
        torch_output = vae_runner.run_torch_vae(
            arguments["hf_model_name"],
            "madebyollin/sdxl-vae-fp16-fix" if arguments["precision"] == "fp16" else "",
            "encode",
            example_input_torch,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 2e-3


def parse_args(args):
    while len(args) > 1:
        if args[0] in arguments.keys():
            arguments[args[0]] = args[1]
        args = args[2:]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parse_args(sys.argv[1:])
    print("Test Config:", arguments)
    unittest.main()
