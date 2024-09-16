# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
import shutil
from transformers import CLIPTokenizer
from turbine_models.custom_models.sd_inference.utils import create_safe_name
from turbine_models.custom_models.sd_inference import schedulers, vae, vae_runner
from turbine_models.custom_models.sdxl_inference import (
    sdxl_prompt_encoder,
    sdxl_prompt_encoder_runner,
    unet,
    unet_runner,
)
from turbine_models.utils.sdxl_benchmark import run_benchmark
import unittest
from tqdm.auto import tqdm
from PIL import Image
import os
import numpy as np
import time
import gc


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
    arguments["attn_spec"] = (
        request.config.getoption("--attn_spec")
        if request.config.getoption("attn_spec")
        else {
            "text_encoder": request.config.getoption("clip_spec"),
            "unet": request.config.getoption("unet_spec"),
            "vae": request.config.getoption("vae_spec"),
        }
    )
    arguments["device"] = request.config.getoption("--device")
    arguments["rt_device"] = request.config.getoption("--rt_device")
    arguments["iree_target_triple"] = request.config.getoption("--iree_target_triple")
    arguments["ireec_flags"] = request.config.getoption("--ireec_flags")
    arguments["attn_flags"] = request.config.getoption("--attn_flags")
    arguments["in_channels"] = int(request.config.getoption("--in_channels"))
    arguments["benchmark"] = request.config.getoption("--benchmark")
    arguments["tracy_profile"] = request.config.getoption("--tracy_profile")
    arguments["compiled_pipeline"] = request.config.getoption("--compiled_pipeline")


@pytest.mark.usefixtures("command_line_args")
class StableDiffusionXLTest(unittest.TestCase):
    def setUp(self):
        from turbine_models.custom_models.sd_inference.sd_pipeline import (
            SharkSDPipeline,
        )

        self.safe_model_name = create_safe_name(arguments["hf_model_name"], "")
        decomp_attn = {
            "text_encoder": True,
            "unet": False,
            "vae": True,
        }
        self.pipe = SharkSDPipeline(
            arguments["hf_model_name"],
            arguments["height"],
            arguments["width"],
            arguments["batch_size"],
            arguments["max_length"],
            arguments["precision"],
            arguments["device"],
            arguments["iree_target_triple"],
            ireec_flags=None,
            attn_spec=arguments["attn_spec"],
            decomp_attn=decomp_attn,
            pipeline_dir="test_vmfbs",
            external_weights_dir="test_weights",
            external_weights=arguments["external_weights"],
            num_inference_steps=arguments["num_inference_steps"],
            cpu_scheduling=True,
            scheduler_id=arguments["scheduler_id"],
            shift=None,
            use_i8_punet=False,
            vae_harness=False,
        )
        self.pipe.prepare_all()

    def test01_PromptEncoder(self):
        if arguments["device"] in ["vulkan", "cuda"]:
            self.skipTest("Compilation error on vulkan; To be tested on cuda.")
        arguments["vmfb_path"] = self.pipe.map["text_encoder"]["vmfb"]
        arguments["external_weight_path"] = self.pipe.map["text_encoder"]["weights"]
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            arguments["hf_model_name"],
            subfolder="tokenizer",
            token=arguments["hf_auth_token"],
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            arguments["hf_model_name"],
            subfolder="tokenizer_2",
            token=arguments["hf_auth_token"],
        )
        (
            text_input_ids_list,
            uncond_input_ids_list,
        ) = sdxl_prompt_encoder_runner.run_tokenize(
            tokenizer_1,
            tokenizer_2,
            arguments["prompt"],
            arguments["negative_prompt"],
            arguments["max_length"],
        )
        (
            turbine_output1,
            turbine_output2,
        ) = sdxl_prompt_encoder_runner.run_prompt_encoder(
            arguments["vmfb_path"],
            self.pipe.map["text_encoder"]["driver"],
            arguments["external_weight_path"],
            text_input_ids_list,
            uncond_input_ids_list,
        )
        torch_model = sdxl_prompt_encoder.PromptEncoderModule(
            arguments["hf_model_name"],
            arguments["precision"],
            arguments["hf_auth_token"],
        )
        torch_output1, torch_output2 = torch_model.forward(
            *text_input_ids_list, *uncond_input_ids_list
        )
        if arguments["benchmark"] or arguments["tracy_profile"]:
            run_benchmark(
                "prompt_encoder",
                arguments["vmfb_path"],
                arguments["external_weight_path"],
                self.pipe.map["text_encoder"]["driver"],
                max_length=arguments["max_length"],
                tracy_profile=arguments["tracy_profile"],
            )
        rtol = 4e-1
        atol = 4e-1
        np.testing.assert_allclose(torch_output1, turbine_output1, rtol, atol)
        np.testing.assert_allclose(torch_output2, turbine_output2, rtol, atol)

    def test02_ExportUnetModel(self):
        if arguments["device"] in ["vulkan", "cuda"]:
            self.skipTest("Unknown error on vulkan; To be tested on cuda.")

        arguments["vmfb_path"] = self.pipe.map["unet"]["vmfb"]
        arguments["external_weight_path"] = self.pipe.map["unet"]["weights"]

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
        timestep = torch.zeros(1, dtype=dtype)
        prompt_embeds = torch.rand(
            (2 * arguments["batch_size"], arguments["max_length"], 2048),
            dtype=dtype,
        )
        text_embeds = torch.rand(2 * arguments["batch_size"], 1280, dtype=dtype)
        time_ids = torch.zeros(2 * arguments["batch_size"], 6, dtype=dtype)
        guidance_scale = torch.Tensor([arguments["guidance_scale"]]).to(dtype)

        turbine = unet_runner.run_unet(
            self.pipe.map["unet"]["driver"],
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
                self.pipe.map["unet"]["driver"],
                max_length=arguments["max_length"],
                height=arguments["height"],
                width=arguments["width"],
                batch_size=arguments["batch_size"],
                in_channels=arguments["in_channels"],
                precision=arguments["precision"],
                tracy_profile=arguments["tracy_profile"],
            )
        rtol = 4e-2
        atol = 4e-1
        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    def test03_ExportVaeModelDecode(self):
        if arguments["device"] in ["vulkan", "cuda"]:
            self.skipTest("Compilation error on vulkan; To be tested on cuda.")

        arguments["vmfb_path"] = self.pipe.map["vae"]["vmfb"]
        arguments["external_weight_path"] = self.pipe.map["unet"]["weights"]
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
        turbine = vae_runner.run_vae_decode(
            self.pipe.map["vae"]["driver"],
            example_input,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            self.pipe.map["vae"]["weights"],
        )
        torch_output = vae_runner.run_torch_vae(
            arguments["hf_model_name"],
            "decode",
            example_input_torch,
        )
        if arguments["benchmark"] or arguments["tracy_profile"]:
            run_benchmark(
                "vae_decode",
                arguments["vmfb_path"],
                arguments["external_weight_path"],
                self.pipe.map["vae"]["driver"],
                height=arguments["height"],
                width=arguments["width"],
                precision=arguments["precision"],
                tracy_profile=arguments["tracy_profile"],
            )
        rtol = 4e-2
        atol = 4e-1
        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    @pytest.mark.xfail(reason="NaN output on rocm, needs triage and file")
    def test04_ExportVaeModelEncode(self):
        if arguments["device"] in ["cpu", "vulkan", "cuda", "rocm"]:
            self.skipTest(
                "Compilation error on cpu, vulkan and rocm; To be tested on cuda."
            )
        arguments["vmfb_path"] = self.pipe.map["vae"]["vmfb"]
        arguments["external_weight_path"] = self.pipe.map["vae"]["weights"]
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
        turbine = vae_runner.run_vae_encode(
            self.pipe.map["vae"]["driver"],
            example_input,
            arguments["vmfb_path"],
            arguments["hf_model_name"],
            self.pipe.map["vae"]["weights"],
        )
        torch_output = vae_runner.run_torch_vae(
            arguments["hf_model_name"],
            "encode",
            example_input_torch,
        )
        if arguments["benchmark"] or arguments["tracy_profile"]:
            run_benchmark(
                "vae_encode",
                arguments["vmfb_path"],
                self.pipe.map["vae"]["weights"],
                self.pipe.map["vae"]["driver"],
                height=arguments["height"],
                width=arguments["width"],
                precision=arguments["precision"],
                tracy_profile=arguments["tracy_profile"],
            )
        rtol = 4e-2
        atol = 4e-2
        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    def test05_t2i_generate_images(self):
        if arguments["device"] in ["vulkan", "cuda"]:
            self.skipTest("Have issues with submodels on vulkan, cuda")

        self.pipe.load_map()
        output = self.pipe.generate_images(
            arguments["prompt"],
            arguments["negative_prompt"],
            arguments["num_inference_steps"],
            1,  # batch count
            arguments["guidance_scale"],
            arguments["seed"],
            True,
            arguments["scheduler_id"],
            True,  # return_img
        )
        assert output is not None

    def test06_t2i_generate_images_punet(self):
        if arguments["device"] in ["vulkan", "cuda"]:
            self.skipTest("Have issues with submodels on vulkan, cuda")
        if getattr(self.pipe, "unet"):
            self.pipe.unload_submodel("unet")
        self.pipe.use_punet = True
        self.pipe.use_i8_punet = True
        self.pipe.setup_punet()
        self.pipe.prepare_all()
        self.pipe.load_map()
        output = self.pipe.generate_images(
            arguments["prompt"],
            arguments["negative_prompt"],
            arguments["num_inference_steps"],
            1,  # batch count
            arguments["guidance_scale"],
            arguments["seed"],
            True,
            arguments["scheduler_id"],
            True,  # return_img
        )
        assert output is not None

    def tearDown(self):
        del self.pipe
        gc.collect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
