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
import copy
import platform
from turbine_models.turbine_tank import turbine_tank


default_arguments = {
    "hf_auth_token": None,
    "hf_model_name": "CompVis/stable-diffusion-v1-4",
    "safe_model_name": "stable-diffusion_v1_4",
    "scheduler_id": "PNDM",
    "num_inference_steps": 5,
    "batch_size": 1,
    "height": 512,
    "width": 512,
    "precision": "fp32",
    "max_length": 77,
    "guidance_scale": 7.5,
    "run_vmfb": True,
    "compile_to": None,
    "external_weight_path": "",
    "vmfb_path": "",
    "external_weights": None,
    "device": "cpu",
    "rt_device": "local-task",
    "iree_target_triple": "x86_64-linux-gnu",
    "vulkan_max_allocation": "4294967296",
    "prompt": "a photograph of an astronaut riding a horse",
    "in_channels": 4,
}
UPLOAD_IR = os.environ.get("TURBINE_TANK_ACTION", "not_upload") == "upload"


unet_model = unet.UnetModel(
    # This is a public model, so no auth required
    default_arguments["hf_model_name"],
)

vae_model = vae.VaeModel(
    # This is a public model, so no auth required
    default_arguments["hf_model_name"],
    custom_vae=None,
)

schedulers_dict = utils.get_schedulers(
    # This is a public model, so no auth required
    "CompVis/stable-diffusion-v1-4",
)
scheduler = schedulers_dict[default_arguments["scheduler_id"]]
scheduler_module = schedulers.Scheduler(
    "CompVis/stable-diffusion-v1-4", default_arguments["num_inference_steps"], scheduler
)


# TODO: this is a mess, don't share args across tests, create a copy for each test
class StableDiffusionTest(unittest.TestCase):
    def testExportT5Model(self):
        current_args = copy.deepcopy(default_arguments)
        current_args["hf_model_name"] = "google/t5-v1_1-small"
        safe_prefix = "t5_v1_1_small"
        blob_name = clip.export_clip_model(
            hf_model_name=current_args["hf_model_name"],
            hf_auth_token=None,
            compile_to="vmfb",
            external_weights=None,
            external_weight_path=None,
            device="cpu",
            target_triple=None,
            max_alloc=None,
            upload_ir=UPLOAD_IR,
        )
        current_args["vmfb_path"] = safe_prefix + "_clip.vmfb"
        turbine = clip_runner.run_clip(
            current_args["rt_device"],
            current_args["prompt"],
            current_args["vmfb_path"],
            current_args["hf_model_name"],
            current_args["hf_auth_token"],
            None,
        )
        torch_output = clip_runner.run_torch_clip(
            current_args["hf_model_name"],
            current_args["hf_auth_token"],
            current_args["prompt"],
        )
        err = utils.largest_error(torch_output, turbine[0])
        assert err < 9e-4
        if UPLOAD_IR:
            new_blob_name = blob_name.split(".")
            new_blob_name = new_blob_name[0] + "-pass.mlir"
            turbine_tank.changeBlobName(blob_name, new_blob_name)
        del current_args
        del turbine

    def testExportClipVitLarge14(self):
        current_args = copy.deepcopy(default_arguments)
        current_args["hf_model_name"] = "openai/clip-vit-large-patch14"
        safe_prefix = "clip_vit_large_patch14"
        blob_name = clip.export_clip_model(
            hf_model_name=current_args["hf_model_name"],
            hf_auth_token=None,
            compile_to="vmfb",
            external_weights="safetensors",
            external_weight_path=safe_prefix + ".safetensors",
            device="cpu",
            target_triple=None,
            max_alloc=None,
            upload_ir=UPLOAD_IR,
        )
        current_args["external_weight_path"] = safe_prefix + ".safetensors"
        current_args["vmfb_path"] = safe_prefix + "_clip.vmfb"
        turbine = clip_runner.run_clip(
            current_args["rt_device"],
            current_args["prompt"],
            current_args["vmfb_path"],
            current_args["hf_model_name"],
            current_args["hf_auth_token"],
            current_args["external_weight_path"],
        )
        torch_output = clip_runner.run_torch_clip(
            current_args["hf_model_name"],
            current_args["hf_auth_token"],
            current_args["prompt"],
        )
        err = utils.largest_error(torch_output, turbine[0])
        assert err < 9e-5
        if UPLOAD_IR:
            new_blob_name = blob_name.split(".")
            new_blob_name = new_blob_name[0] + "-pass.mlir"
            turbine_tank.changeBlobName(blob_name, new_blob_name)
        if platform.system() != "Windows":
            os.remove(current_args["external_weight_path"])
            os.remove(current_args["vmfb_path"])
        del current_args
        del turbine

    def testExportClipModel(self):
        current_args = copy.deepcopy(default_arguments)
        current_args["hf_model_name"] = "CompVis/stable-diffusion-v1-4"
        blob_name = clip.export_clip_model(
            # This is a public model, so no auth required
            "CompVis/stable-diffusion-v1-4",
            None,
            "vmfb",
            "safetensors",
            "stable_diffusion_v1_4_clip.safetensors",
            "cpu",
            upload_ir=UPLOAD_IR,
        )
        current_args["external_weight_path"] = "stable_diffusion_v1_4_clip.safetensors"
        current_args["vmfb_path"] = "stable_diffusion_v1_4_clip.vmfb"
        turbine = clip_runner.run_clip(
            current_args["rt_device"],
            current_args["prompt"],
            current_args["vmfb_path"],
            current_args["hf_model_name"],
            current_args["hf_auth_token"],
            current_args["external_weight_path"],
        )
        torch_output = clip_runner.run_torch_clip(
            current_args["hf_model_name"],
            current_args["hf_auth_token"],
            current_args["prompt"],
        )
        err = utils.largest_error(torch_output, turbine[0])
        assert err < 9e-5
        if UPLOAD_IR:
            new_blob_name = blob_name.split(".")
            new_blob_name = new_blob_name[0] + "-pass.mlir"
            turbine_tank.changeBlobName(blob_name, new_blob_name)
        if platform.system() != "Windows":
            os.remove(current_args["external_weight_path"])
            os.remove(current_args["vmfb_path"])
        del current_args
        del turbine

    @unittest.expectedFailure
    def testExportUnetModel(self):
        current_args = copy.deepcopy(default_arguments)
        blob_name = unet.export_unet_model(
            unet_model,
            "CompVis/stable-diffusion-v1-4",
            current_args["batch_size"],
            current_args["height"],
            current_args["width"],
            current_args["precision"],
            current_args["max_length"],
            None,
            "vmfb",
            "safetensors",
            "stable_diffusion_v1_4_unet.safetensors",
            "cpu",
            upload_ir=UPLOAD_IR,
        )
        current_args["external_weight_path"] = "stable_diffusion_v1_4_unet.safetensors"
        current_args["vmfb_path"] = "stable_diffusion_v1_4_unet.vmfb"
        sample = torch.rand(
            current_args["batch_size"],
            current_args["in_channels"],
            current_args["height"] // 8,
            current_args["width"] // 8,
            dtype=torch.float32,
        )

        timestep = torch.zeros(1, dtype=torch.float32)
        if current_args["hf_model_name"] == "CompVis/stable-diffusion-v1-4":
            encoder_hidden_states = torch.rand(2, 77, 768, dtype=torch.float32)
        elif current_args["hf_model_name"] == "stabilityai/stable-diffusion-2-1-base":
            encoder_hidden_states = torch.rand(2, 77, 1024, dtype=torch.float32)
        guidance_scale = torch.tensor(
            [current_args["guidance_scale"]], dtype=torch.float32
        )

        turbine = unet_runner.run_unet(
            current_args["rt_device"],
            sample,
            timestep,
            encoder_hidden_states,
            guidance_scale,
            current_args["vmfb_path"],
            current_args["hf_model_name"],
            current_args["hf_auth_token"],
            current_args["external_weight_path"],
        )
        torch_output = unet_runner.run_torch_unet(
            current_args["hf_model_name"],
            current_args["hf_auth_token"],
            sample,
            timestep,
            encoder_hidden_states,
            guidance_scale,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-5
        if UPLOAD_IR:
            new_blob_name = blob_name.split(".")
            new_blob_name = new_blob_name[0] + "-pass.mlir"
            turbine_tank.changeBlobName(blob_name, new_blob_name)
        os.remove("stable_diffusion_v1_4_unet.safetensors")
        os.remove("stable_diffusion_v1_4_unet.vmfb")
        del torch_output
        del turbine

    def testExportVaeModelDecode(self):
        current_args = copy.deepcopy(default_arguments)
        blob_name = vae.export_vae_model(
            vae_model,
            # This is a public model, so no auth required
            "CompVis/stable-diffusion-v1-4",
            current_args["batch_size"],
            current_args["height"],
            current_args["width"],
            None,
            "vmfb",
            "safetensors",
            "stable_diffusion_v1_4_vae.safetensors",
            "cpu",
            variant="decode",
            upload_ir=UPLOAD_IR,
        )
        current_args["external_weight_path"] = "stable_diffusion_v1_4_vae.safetensors"
        current_args["vmfb_path"] = "stable_diffusion_v1_4_vae.vmfb"
        example_input = torch.rand(
            current_args["batch_size"],
            4,
            current_args["height"] // 8,
            current_args["width"] // 8,
            dtype=torch.float32,
        )
        turbine = vae_runner.run_vae(
            current_args["rt_device"],
            example_input,
            current_args["vmfb_path"],
            current_args["hf_model_name"],
            current_args["external_weight_path"],
        )
        torch_output = vae_runner.run_torch_vae(
            current_args["hf_model_name"],
            "decode",
            example_input,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-5
        if UPLOAD_IR:
            new_blob_name = blob_name.split(".")
            new_blob_name = new_blob_name[0] + "-pass.mlir"
            turbine_tank.changeBlobName(blob_name, new_blob_name)
        del current_args
        del torch_output
        del turbine
        os.remove("stable_diffusion_v1_4_vae.safetensors")
        os.remove("stable_diffusion_v1_4_vae.vmfb")

    def testExportVaeModelEncode(self):
        current_args = copy.deepcopy(default_arguments)
        blob_name = vae.export_vae_model(
            vae_model,
            # This is a public model, so no auth required
            "CompVis/stable-diffusion-v1-4",
            current_args["batch_size"],
            current_args["height"],
            current_args["width"],
            current_args["precision"],
            "vmfb",
            "safetensors",
            "stable_diffusion_v1_4_vae.safetensors",
            "cpu",
            variant="encode",
            upload_ir=UPLOAD_IR,
        )
        current_args["external_weight_path"] = "stable_diffusion_v1_4_vae.safetensors"
        current_args["vmfb_path"] = "stable_diffusion_v1_4_vae.vmfb"
        example_input = torch.rand(
            current_args["batch_size"],
            3,
            current_args["height"],
            current_args["width"],
            dtype=torch.float32,
        )
        turbine = vae_runner.run_vae(
            current_args["rt_device"],
            example_input,
            current_args["vmfb_path"],
            current_args["hf_model_name"],
            current_args["external_weight_path"],
        )
        torch_output = vae_runner.run_torch_vae(
            current_args["hf_model_name"],
            "encode",
            example_input,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 3e-3
        if UPLOAD_IR:
            new_blob_name = blob_name.split(".")
            new_blob_name = new_blob_name[0] + "-pass.mlir"
            turbine_tank.changeBlobName(blob_name, new_blob_name)
        os.remove("stable_diffusion_v1_4_vae.safetensors")
        os.remove("stable_diffusion_v1_4_vae.vmfb")
        del current_args
        del turbine

    @unittest.expectedFailure
    def testExportPNDMScheduler(self):
        current_args = copy.deepcopy(default_arguments)
        safe_name = "stable_diffusion_v1_4_scheduler"
        blob_name = schedulers.export_scheduler(
            scheduler_module,
            # This is a public model, so no auth required
            "CompVis/stable-diffusion-v1-4",
            current_args["batch_size"],
            current_args["height"],
            current_args["width"],
            None,
            "vmfb",
            "safetensors",
            "stable_diffusion_v1_4_scheduler.safetensors",
            "cpu",
            upload_ir=UPLOAD_IR,
        )
        current_args["external_weight_path"] = safe_name + ".safetensors"
        current_args["vmfb_path"] = safe_name + ".vmfb"
        sample = torch.rand(
            current_args["batch_size"],
            4,
            current_args["height"] // 8,
            current_args["width"] // 8,
            dtype=torch.float32,
        )
        encoder_hidden_states = torch.rand(2, 77, 768, dtype=torch.float32)
        turbine = schedulers_runner.run_scheduler(
            current_args["rt_device"],
            sample,
            encoder_hidden_states,
            current_args["vmfb_path"],
            current_args["hf_model_name"],
            current_args["hf_auth_token"],
            current_args["external_weight_path"],
        )
        torch_output = schedulers_runner.run_torch_scheduler(
            current_args["hf_model_name"],
            scheduler,
            current_args["num_inference_steps"],
            sample,
            encoder_hidden_states,
        )
        err = utils.largest_error(torch_output, turbine)
        assert err < 9e-3
        if UPLOAD_IR:
            new_blob_name = blob_name.split(".")
            new_blob_name = new_blob_name[0] + "-pass.mlir"
            turbine_tank.changeBlobName(blob_name, new_blob_name)
        os.remove("stable_diffusion_v1_4_scheduler.safetensors")
        os.remove("stable_diffusion_v1_4_scheduler.vmfb")
        del current_args
        del torch_output
        del turbine


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
