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
from PIL import Image
from turbine_models.turbine_tank import turbine_tank


default_arguments = {
    "hf_auth_token": None,
    "hf_model_name": "CompVis/stable-diffusion-v1-4",
    "safe_model_name": "stable-diffusion_v1_4",
    "scheduler_id": "EulerDiscrete",
    "num_inference_steps": 2,
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
    "prompt": "a photograph of an astronaut riding a horse",
    "negative_prompt": "blurry, out of focus",
    "in_channels": 4,
    "vae_decomp_attn": True,
    "seed": 0,
    "use_i8_punet": False,
    "attn_spec": None,
    "cpu_scheduling": True,
}
UPLOAD_IR = os.environ.get("TURBINE_TANK_ACTION", "not_upload") == "upload"


# TODO: this is a mess, don't share args across tests, create a copy for each test
class StableDiffusionTest(unittest.TestCase):
    def testExportClipModel(self):
        current_args = copy.deepcopy(default_arguments)
        current_args["hf_model_name"] = "CompVis/stable-diffusion-v1-4"
        safe_prefix = utils.create_safe_name(
            current_args["hf_model_name"].split("/")[-1], "clip"
        )
        blob_name = clip.export_clip_model(
            hf_model_name=current_args["hf_model_name"],
            max_length=current_args["max_length"],
            precision=current_args["precision"],
            compile_to="vmfb",
            external_weights="safetensors",
            external_weight_path=safe_prefix + ".safetensors",
            device="cpu",
            target=current_args["iree_target_triple"],
            exit_on_vmfb=False,
            upload_ir=UPLOAD_IR,
        )
        current_args["external_weight_path"] = safe_prefix + ".safetensors"
        current_args["vmfb_path"] = blob_name
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

    def testExportUnetModel(self):
        current_args = copy.deepcopy(default_arguments)
        blob_name = unet.export_unet_model(
            hf_model_name=current_args["hf_model_name"],
            batch_size=current_args["batch_size"],
            height=current_args["height"],
            width=current_args["width"],
            precision=current_args["precision"],
            max_length=current_args["max_length"],
            compile_to="vmfb",
            external_weights="safetensors",
            external_weight_path="stable_diffusion_unet.safetensors",
            device="cpu",
            target=current_args["iree_target_triple"],
            upload_ir=UPLOAD_IR,
        )
        current_args["external_weight_path"] = "stable_diffusion_unet.safetensors"
        current_args["vmfb_path"] = blob_name
        sample = torch.rand(
            current_args["batch_size"] * 2,
            current_args["in_channels"],
            current_args["height"] // 8,
            current_args["width"] // 8,
            dtype=torch.float32,
        )

        timestep = torch.zeros(1, dtype=torch.float32)
        if current_args["hf_model_name"] == "CompVis/stable-diffusion-v1-4":
            encoder_hidden_states = torch.rand(
                2, current_args["max_length"], 768, dtype=torch.float32
            )
        elif current_args["hf_model_name"] == "stabilityai/stable-diffusion-2-1-base":
            encoder_hidden_states = torch.rand(
                2, current_args["max_length"], 1024, dtype=torch.float32
            )
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
            "float32",
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
        os.remove("stable_diffusion_unet.safetensors")
        os.remove(blob_name)
        del torch_output
        del turbine

    def testExportVaeModelDecode(self):
        current_args = copy.deepcopy(default_arguments)
        blob_name = vae.export_vae_model(
            hf_model_name=current_args["hf_model_name"],
            batch_size=current_args["batch_size"],
            height=current_args["height"],
            width=current_args["width"],
            precision=current_args["precision"],
            compile_to="vmfb",
            external_weights="safetensors",
            external_weight_path="stable_diffusion_v1_4_vae.safetensors",
            device="cpu",
            target=current_args["iree_target_triple"],
            decomp_attn=current_args["vae_decomp_attn"],
            upload_ir=UPLOAD_IR,
        )
        current_args["external_weight_path"] = "stable_diffusion_v1_4_vae.safetensors"
        current_args["vmfb_path"] = blob_name
        example_input = torch.rand(
            current_args["batch_size"],
            4,
            current_args["height"] // 8,
            current_args["width"] // 8,
            dtype=torch.float32,
        )
        turbine = vae_runner.run_vae_decode(
            current_args["rt_device"],
            example_input,
            current_args["vmfb_path"],
            current_args["hf_model_name"],
            current_args["external_weight_path"],
        )
        torch_output = vae_runner.run_torch_vae_decode(
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
        os.remove(blob_name)

    def testSDPipeline(self):
        from turbine_models.custom_models.sd_inference.sd_pipeline import (
            SharkSDPipeline,
        )

        current_args = copy.deepcopy(default_arguments)
        decomp_attn = {
            "text_encoder": False,
            "unet": False,
            "vae": current_args["vae_decomp_attn"],
        }
        sd_pipe = SharkSDPipeline(
            current_args["hf_model_name"],
            current_args["height"],
            current_args["width"],
            current_args["batch_size"],
            current_args["max_length"],
            current_args["precision"],
            current_args["device"],
            current_args["iree_target_triple"],
            ireec_flags=None,  # ireec_flags
            attn_spec=current_args["attn_spec"],
            decomp_attn=decomp_attn,
            pipeline_dir="test_vmfbs",  # pipeline_dir
            external_weights_dir="test_weights",  # external_weights_dir
            external_weights=current_args["external_weights"],
            num_inference_steps=current_args["num_inference_steps"],
            cpu_scheduling=True,
            scheduler_id=current_args["scheduler_id"],
            shift=None,  # shift
            use_i8_punet=current_args["use_i8_punet"],
        )
        sd_pipe.prepare_all()
        sd_pipe.load_map()
        output = sd_pipe.generate_images(
            current_args["prompt"],
            current_args["negative_prompt"],
            current_args["num_inference_steps"],
            1,  # batch count
            current_args["guidance_scale"],
            current_args["seed"],
            current_args["cpu_scheduling"],
            current_args["scheduler_id"],
            True,  # return_img
        )
        assert output is not None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
