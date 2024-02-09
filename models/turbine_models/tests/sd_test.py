# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import logging
from turbine_models.custom_models.sd_inference import (
    clip,
    clip_runner,
    unet,
    unet_runner,
    vae,
    vae_runner,
    schedulers,
    schedulers_runner,
)
from transformers import CLIPTextModel
from turbine_models.custom_models.sd_inference import utils
import torch
import unittest
import os


arguments = {
    "hf_auth_token": None,
    "hf_model_name": "stabilityai/stable-diffusion-2-1",
    "safe_model_name": "stable_diffusion_2_1",
    "scheduler_id": "PNDM",
    "num_inference_steps": 5,
    "batch_size": 1,
    "height": 512,
    "width": 512,
    "precision": "fp16",
    "max_length": 77,
    "guidance_scale": 7.5,
    "run_vmfb": True,
    "compile_to": None,
    "external_weight_path": "",
    "vmfb_path": "",
    "external_weights": None,
    "device": "local-task",
    "iree_target_triple": "",
    "vulkan_max_allocation": "4294967296",
    "prompt": "a photograph of an astronaut riding a horse",
    "in_channels": 4,
}


unet_model = unet.UnetModel(
    # This is a public model, so no auth required
    arguments["hf_model_name"],
)

vae_model = vae.VaeModel(
    # This is a public model, so no auth required
    arguments["hf_model_name"],
    custom_vae=None,
)

schedulers_dict = utils.get_schedulers(
    # This is a public model, so no auth required
    "CompVis/stable-diffusion-v1-4",
)
scheduler = schedulers_dict[arguments["scheduler_id"]]
scheduler_module = schedulers.Scheduler(
    "CompVis/stable-diffusion-v1-4", arguments["num_inference_steps"], scheduler
)


class StableDiffusionTest(unittest.TestCase):
    def testExportClipModel(self):
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
        with self.assertRaises(SystemExit) as cm:
            clip.export_clip_model(
                # This is a public model, so no auth required
                arguments["hf_model_name"],
                None,
                "vmfb",
                "safetensors",
                f"{arguments['safe_model_name']}_clip.safetensors",
                "cpu",
                upload_ir=upload_ir_var == "upload",
            )
        self.assertEqual(cm.exception.code, None)
        arguments[
            "external_weight_path"
        ] = f"{arguments['safe_model_name']}_clip.safetensors"
        arguments["vmfb_path"] = f"{arguments['safe_model_name']}_clip.vmfb"
        turbine = clip_runner.run_clip(
            arguments["device"],
            arguments["prompt"],
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["external_weight_path"],
        )
        torch_output = clip_runner.run_torch_clip(
            arguments["hf_model_name"], arguments["hf_auth_token"], arguments["prompt"]
        )
        err = utils.largest_error(torch_output, turbine[0])
        assert err < 9e-5
        # os.remove(f"{arguments['safe_model_name']}_clip.safetensors")
        # os.remove(f"{arguments['safe_model_name']}_clip.vmfb")

    def testExportUnetModel(self):
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
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
        dtype = torch.float16 if arguments["precision"] == "fp16" else torch.float32
        sample = torch.rand(
            arguments["batch_size"],
            arguments["in_channels"],
            arguments["height"] // 8,
            arguments["width"] // 8,
            dtype=dtype,
        )
        timestep = torch.zeros(1, dtype=dtype)
        encoder_hidden_states = torch.rand(2, 77, 768, dtype=dtype)
        guidance_scale = torch.Tensor([arguments["guidance_scale"]]).to(dtype)

        turbine = unet_runner.run_unet(
            arguments["device"],
            sample,
            timestep,
            encoder_hidden_states,
            guidance_scale,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["external_weight_path"],
        )
        torch_output = unet_runner.run_torch_unet(
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            sample,
            timestep,
            encoder_hidden_states,
            guidance_scale,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-5
        # os.remove(f"{arguments['safe_model_name']}_unet.safetensors")
        # os.remove(f"{arguments['safe_model_name']}_unet.vmfb")

    def testExportVaeModelDecode(self):
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
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
                external_weight_path=f"{arguments['safe_model_name']}_vae.safetensors",
                device="cpu",
                variant="decode",
                upload_ir=upload_ir_var == "upload",
            )
        self.assertEqual(cm.exception.code, None)
        arguments[
            "external_weight_path"
        ] = f"{arguments['safe_model_name']}_vae.safetensors"
        arguments["vmfb_path"] = f"{arguments['safe_model_name']}_vae.vmfb"
        dtype = torch.float16 if arguments["precision"] == "fp16" else torch.float32
        example_input = torch.rand(
            arguments["batch_size"],
            4,
            arguments["height"] // 8,
            arguments["width"] // 8,
            dtype=dtype,
        )
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
            example_input,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-5
        # os.remove(f"{arguments['safe_model_name']}_vae.safetensors")
        # os.remove(f"{arguments['safe_model_name']}_vae.vmfb")

    def testExportVaeModelEncode(self):
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
        with self.assertRaises(SystemExit) as cm:
            vae.export_vae_model(
                vae_model,
                # This is a public model, so no auth required
                arguments["hf_model_name"],
                arguments["batch_size"],
                arguments["height"],
                arguments["width"],
                arguments["precision"],
                "vmfb",
                external_weights="safetensors",
                external_weight_path=f"{arguments['safe_model_name']}_vae.safetensors",
                device="cpu",
                variant="encode",
                upload_ir=upload_ir_var == "upload",
            )
        self.assertEqual(cm.exception.code, None)
        arguments[
            "external_weight_path"
        ] = f"{arguments['safe_model_name']}_vae.safetensors"
        arguments["vmfb_path"] = f"{arguments['safe_model_name']}_vae.vmfb"
        dtype = torch.float16 if arguments["precision"] == "fp16" else torch.float32
        example_input = torch.rand(
            arguments["batch_size"],
            3,
            arguments["height"],
            arguments["width"],
            dtype=dtype,
        )
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
            example_input,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 3e-3
        os.remove("stable_diffusion_v1_4_vae.safetensors")
        os.remove("stable_diffusion_v1_4_vae.vmfb")

    @unittest.expectedFailure
    def testExportPNDMScheduler(self):
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
        with self.assertRaises(SystemExit) as cm:
            schedulers.export_scheduler(
                scheduler_module,
                # This is a public model, so no auth required
                "CompVis/stable-diffusion-v1-4",
                arguments["batch_size"],
                arguments["height"],
                arguments["width"],
                None,
                "vmfb",
                "safetensors",
                "stable_diffusion_v1_4_scheduler.safetensors",
                "cpu",
                upload_ir=upload_ir_var == "upload",
            )
        self.assertEqual(cm.exception.code, None)
        arguments[
            "external_weight_path"
        ] = "stable_diffusion_v1_4_scheduler.safetensors"
        arguments["vmfb_path"] = "stable_diffusion_v1_4_scheduler.vmfb"
        sample = torch.rand(
            arguments["batch_size"],
            4,
            arguments["height"] // 8,
            arguments["width"] // 8,
            dtype=torch.float32,
        )
        encoder_hidden_states = torch.rand(2, 77, 768, dtype=torch.float32)
        turbine = schedulers_runner.run_scheduler(
            arguments["device"],
            sample,
            encoder_hidden_states,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["external_weight_path"],
        )
        torch_output = schedulers_runner.run_torch_scheduler(
            arguments["hf_model_name"],
            scheduler,
            arguments["num_inference_steps"],
            sample,
            encoder_hidden_states,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-3
        os.remove("stable_diffusion_v1_4_scheduler.safetensors")
        os.remove("stable_diffusion_v1_4_scheduler.vmfb")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
