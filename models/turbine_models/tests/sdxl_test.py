# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import logging
from turbine_models.custom_models.sdxl_inference import (
    clip,
    clip_runner,
    unet,
    unet_runner,
    vae,
    vae_runner,
)
from transformers import CLIPTextModel
from turbine_models.custom_models.sd_inference import utils
import torch
import unittest
import os


arguments = {
    "hf_auth_token": None,
    "hf_model_name": "stabilityai/sdxl-turbo",
    "safe_model_name": "sdxl_turbo",
    "batch_size": 1,
    "height": 512,
    "width": 512,
    "precision": "f16",
    "max_length": 77,
    "guidance_scale": 7.5,
    "run_vmfb": True,
    "compile_to": None,
    "external_weight_path": "",
    "vmfb_path": "",
    "external_weights": "safetensors",
    "device": "local-task",
    "iree_target_triple": "",
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
    custom_vae="madebyollin/sdxl-vae-fp16-fix",
)


class StableDiffusionTest(unittest.TestCase):
    def test01_ExportClipModels(self):
        vmfb_path_1, vmfb_path_2, _, _, = clip.export_clip_model(
            # This is a public model, so no auth required
            arguments["hf_model_name"],
            None,
            "vmfb",
            "safetensors",
            f"{arguments['safe_model_name']}" + "_clip",
            "cpu",
        )
        assert os.path.exists(f"{arguments['safe_model_name']}_clip_1.vmfb")
        assert os.path.exists(f"{arguments['safe_model_name']}_clip_2.vmfb")
        arguments["external_weight_path_1"] = f"{arguments['safe_model_name']}_clip_1.safetensors"
        arguments["external_weight_path_2"] = f"{arguments['safe_model_name']}_clip_2.safetensors"
        arguments["vmfb_path_1"] = vmfb_path_1
        arguments["vmfb_path_2"] = vmfb_path_2
        turbine_1 = clip_runner.run_clip(
            arguments["device"],
            arguments["prompt"],
            arguments["vmfb_path_1"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["external_weight_path_1"],
            index=1,
        )
        turbine_2 = clip_runner.run_clip(
            arguments["device"],
            arguments["prompt"],
            arguments["vmfb_path_2"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["external_weight_path_2"],
            index=2,
        )
        torch_output_1, torch_output_2 = clip_runner.run_torch_clip(
            arguments["hf_model_name"], arguments["hf_auth_token"], arguments["prompt"], arguments["precision"],
        )
        err1 = utils.largest_error(torch_output_1, turbine_1[0])
        err2 = utils.largest_error(torch_output_2, turbine_2[0])
        assert err1 < 9e-5 and err2 < 9e-5

    # def test02_ExportClipModelBreakdown(self):
    #     os.remove(f"{arguments['safe_model_name']}_clip_1.safetensors")
    #     os.remove(f"{arguments['safe_model_name']}_clip_1.vmfb")
    #     os.remove(f"{arguments['safe_model_name']}_clip_2.safetensors")
    #     os.remove(f"{arguments['safe_model_name']}_clip_2.vmfb")

    def test03_ExportUnetModel(self):
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
                device="cpu",
            )
        self.assertEqual(cm.exception.code, None)
        arguments[
            "external_weight_path"
        ] = f"{arguments['safe_model_name']}_unet.safetensors"
        arguments["vmfb_path"] = f"{arguments['safe_model_name']}_unet.vmfb"
        dtype = torch.float16 if arguments["precision"] == "f16" else torch.float32
        sample = torch.rand(
            (
                arguments["batch_size"],
                arguments["in_channels"],
                arguments["height"] // 8,
                arguments["width"] // 8
            ),
            dtype=dtype,
        )
        timestep = torch.zeros((1), dtype=torch.int64)
        prompt_embeds = torch.rand(
            (2 * arguments["batch_size"], arguments["max_length"], 2048), dtype=dtype
        )
        text_embeds = torch.rand((2 * arguments["batch_size"], 1280), dtype=dtype)
        time_ids = torch.zeros((2 * arguments["batch_size"], 6), dtype=dtype)
        guidance_scale = torch.Tensor([arguments["guidance_scale"]]).to(dtype)

        turbine = unet_runner.run_unet(
            arguments["device"],
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
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-5

    # def test04_ExportUnetModelBreakdown(self):
    #     os.remove(f"{arguments['safe_model_name']}_unet.safetensors")
    #     os.remove(f"{arguments['safe_model_name']}_unet.vmfb")


    def test05_ExportVaeModelDecode(self):
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
                external_weight_path=f"{arguments['safe_model_name']}_vae_decode.safetensors",
                device="cpu",
                variant="decode",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path"] = f"{arguments['safe_model_name']}_vae_decode.safetensors"
        arguments["vmfb_path"] = f"{arguments['safe_model_name']}_vae_decode.vmfb"
        example_input = torch.rand(
            arguments["batch_size"],
            4,
            arguments["height"] // 8,
            arguments["width"] // 8,
            dtype=torch.float32,
        )
        example_input_torch = example_input
        if arguments["precision"] == "f16":
            example_input = example_input.half()
        turbine = vae_runner.run_vae(
            arguments["device"],
            example_input,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["external_weight_path"],
        )
        torch_output = vae_runner.run_torch_vae(
            arguments["hf_model_name"],
            "decode",
            example_input_torch,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-5

    def test06_ExportVaeModelEncode(self): 
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
                external_weight_path=f"{arguments['safe_model_name']}_vae_encode.safetensors",
                device="cpu",
                variant="encode",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path"] = f"{arguments['safe_model_name']}_vae_encode.safetensors"
        arguments["vmfb_path"] = f"{arguments['safe_model_name']}_vae_encode.vmfb"
        example_input = torch.rand(
            arguments["batch_size"],
            3,
            arguments["height"],
            arguments["width"],
            dtype=torch.float32,
        )
        example_input_torch = example_input
        if arguments["precision"] == "f16":
            example_input = example_input.half()
        turbine = vae_runner.run_vae(
            arguments["device"],
            example_input,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["external_weight_path"],
        )
        torch_output = vae_runner.run_torch_vae(
            arguments["hf_model_name"],
            "encode",
            example_input_torch,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 2e-3

    # def test07_ExportVaeModelBreakdown(self):
    #     os.remove(f"{arguments['safe_model_name']}_vae.safetensors")
    #     os.remove(f"{arguments['safe_model_name']}_vae.vmfb")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
