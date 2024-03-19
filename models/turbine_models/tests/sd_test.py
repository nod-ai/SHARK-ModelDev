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
    "hf_model_name": "CompVis/stable-diffusion-v1-4",
    "scheduler_id": "PNDM",
    "num_inference_steps": 5,
    "batch_size": 1,
    "height": 512,
    "width": 512,
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
    "CompVis/stable-diffusion-v1-4",
    None,
)

vae_model = vae.VaeModel(
    # This is a public model, so no auth required
    "CompVis/stable-diffusion-v1-4",
    None,
)

schedulers_dict = utils.get_schedulers(
    # This is a public model, so no auth required
    "CompVis/stable-diffusion-v1-4",
)
scheduler = schedulers_dict[arguments["scheduler_id"]]
scheduler_module = schedulers.Scheduler(
    "CompVis/stable-diffusion-v1-4", arguments["num_inference_steps"], scheduler
)

# TODO: this is a mess, don't share args across tests, create a copy for each test
class StableDiffusionTest(unittest.TestCase):
    def testExportT5Model(self):
        arguments["hf_model_name"] = "google/t5-v1_1-small"
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
        safe_prefix = "google_t5-v1_1-small"
        with self.assertRaises(SystemExit) as cm:
            clip.export_clip_model(
                hf_model_name=arguments["hf_model_name"],
                hf_auth_token=None,
                compile_to="vmfb",
                external_weights=None,
                external_weight_path=None,
                device="cpu",
                target_triple=None,
                max_alloc=None,
                upload_ir=upload_ir_var == "upload",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["vmfb_path"] = safe_prefix + ".vmfb"
        turbine = clip_runner.run_clip(
            arguments["device"],
            arguments["prompt"],
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            None,
        )
        torch_output = clip_runner.run_torch_clip(
            arguments["hf_model_name"], arguments["hf_auth_token"], arguments["prompt"]
        )
        err = utils.largest_error(torch_output, turbine[0])
        assert err < 9e-5
        os.remove(safe_prefix + ".vmfb")

    def testExportClipVitLarge14(self):
        arguments["hf_model_name"] = "openai/clip-vit-large-patch14"
        safe_prefix = "openai_clip_vit_large_patch14"
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
        with self.assertRaises(SystemExit) as cm:
            clip.export_clip_model(
                hf_model_name=arguments["hf_model_name"],
                hf_auth_token=None,
                compile_to="vmfb",
                external_weights="safetensors",
                external_weight_path=safe_prefix + ".safetensors",
                device="cpu",
                target_triple=None,
                max_alloc=None,
                upload_ir=upload_ir_var == "upload",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path"] = safe_prefix + ".safetensors"
        arguments["vmfb_path"] = safe_prefix + ".vmfb"
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
        os.remove(safe_prefix + ".safetensors")
        os.remove(safe_prefix + ".vmfb")

    def testExportClipModel(self):
        arguments["hf_model_name"] = "CompVis/stable-diffusion-v1-4"
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
        with self.assertRaises(SystemExit) as cm:
            clip.export_clip_model(
                # This is a public model, so no auth required
                "CompVis/stable-diffusion-v1-4",
                None,
                "vmfb",
                "safetensors",
                "stable_diffusion_v1_4_clip.safetensors",
                "cpu",
                upload_ir=upload_ir_var == "upload",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path"] = "stable_diffusion_v1_4_clip.safetensors"
        arguments["vmfb_path"] = "stable_diffusion_v1_4_clip.vmfb"
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
        os.remove("stable_diffusion_v1_4_clip.safetensors")
        os.remove("stable_diffusion_v1_4_clip.vmfb")

    def testExportUnetModel(self):
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
        with self.assertRaises(SystemExit) as cm:
            unet.export_unet_model(
                unet_model,
                # This is a public model, so no auth required
                "CompVis/stable-diffusion-v1-4",
                arguments["batch_size"],
                arguments["height"],
                arguments["width"],
                None,
                "vmfb",
                "safetensors",
                "stable_diffusion_v1_4_unet.safetensors",
                "cpu",
                upload_ir=upload_ir_var == "upload",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path"] = "stable_diffusion_v1_4_unet.safetensors"
        arguments["vmfb_path"] = "stable_diffusion_v1_4_unet.vmfb"
        sample = torch.rand(
            arguments["batch_size"],
            arguments["in_channels"],
            arguments["height"] // 8,
            arguments["width"] // 8,
            dtype=torch.float32,
        )
        timestep = torch.zeros(1, dtype=torch.float32)
        encoder_hidden_states = torch.rand(2, 77, 768, dtype=torch.float32)

        turbine = unet_runner.run_unet(
            arguments["device"],
            sample,
            timestep,
            encoder_hidden_states,
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
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-5
        os.remove("stable_diffusion_v1_4_unet.safetensors")
        os.remove("stable_diffusion_v1_4_unet.vmfb")

    def testExportVaeModelDecode(self):
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
        with self.assertRaises(SystemExit) as cm:
            vae.export_vae_model(
                vae_model,
                # This is a public model, so no auth required
                "CompVis/stable-diffusion-v1-4",
                arguments["batch_size"],
                arguments["height"],
                arguments["width"],
                None,
                "vmfb",
                "safetensors",
                "stable_diffusion_v1_4_vae.safetensors",
                "cpu",
                variant="decode",
                upload_ir=upload_ir_var == "upload",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path"] = "stable_diffusion_v1_4_vae.safetensors"
        arguments["vmfb_path"] = "stable_diffusion_v1_4_vae.vmfb"
        example_input = torch.rand(
            arguments["batch_size"],
            4,
            arguments["height"] // 8,
            arguments["width"] // 8,
            dtype=torch.float32,
        )
        turbine = vae_runner.run_vae(
            arguments["device"],
            example_input,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["external_weight_path"],
        )
        torch_output = vae_runner.run_torch_vae(
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            "decode",
            example_input,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-5
        os.remove("stable_diffusion_v1_4_vae.safetensors")
        os.remove("stable_diffusion_v1_4_vae.vmfb")

    # https://github.com/nod-ai/SHARK-Turbine/issues/536
    @unittest.expectedFailure
    def testExportVaeModelEncode(self):
        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")
        with self.assertRaises(SystemExit) as cm:
            vae.export_vae_model(
                vae_model,
                # This is a public model, so no auth required
                "CompVis/stable-diffusion-v1-4",
                arguments["batch_size"],
                arguments["height"],
                arguments["width"],
                None,
                "vmfb",
                "safetensors",
                "stable_diffusion_v1_4_vae.safetensors",
                "cpu",
                variant="encode",
                upload_ir=upload_ir_var == "upload",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_path"] = "stable_diffusion_v1_4_vae.safetensors"
        arguments["vmfb_path"] = "stable_diffusion_v1_4_vae.vmfb"
        example_input = torch.rand(
            arguments["batch_size"],
            3,
            arguments["height"],
            arguments["width"],
            dtype=torch.float32,
        )
        turbine = vae_runner.run_vae(
            arguments["device"],
            example_input,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
            arguments["external_weight_path"],
        )
        torch_output = vae_runner.run_torch_vae(
            arguments["hf_model_name"],
            arguments["hf_auth_token"],
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
