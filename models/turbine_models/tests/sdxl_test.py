# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
from transformers import CLIPTokenizer
from turbine_models.custom_models.sd_inference.utils import create_safe_name
from turbine_models.custom_models.sd_inference import schedulers, vae
from turbine_models.custom_models.sdxl_inference import (
    sdxl_prompt_encoder,
    sdxl_prompt_encoder_runner,
    unet,
    unet_runner,
    sdxl_scheduled_unet,
    sdxl_scheduled_unet_runner,
    vae_runner,
    sdxl_compiled_pipeline,
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
    arguments["attn_spec"] = request.config.getoption("--attn_spec")
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
        self.safe_model_name = create_safe_name(arguments["hf_model_name"], "")

    def test01_ExportPromptEncoder(self):
        if arguments["device"] in ["vulkan", "cuda"]:
            self.skipTest(
                "Compilation error on vulkan; Runtime error on rocm; To be tested on cuda."
            )
        arguments["external_weight_path"] = (
            "prompt_encoder." + arguments["external_weights"]
        )
        prompt_encoder_vmfb = sdxl_prompt_encoder.export_prompt_encoder(
            arguments["hf_model_name"],
            hf_auth_token=None,
            max_length=arguments["max_length"],
            batch_size=arguments["batch_size"],
            precision=arguments["precision"],
            compile_to="vmfb",
            external_weights="safetensors",
            external_weight_path=arguments["external_weight_path"],
            device=arguments["device"],
            target=arguments["iree_target_triple"],
        )
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
            prompt_encoder_vmfb,
            arguments["rt_device"],
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
                prompt_encoder_vmfb,
                arguments["external_weight_path"],
                arguments["rt_device"],
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
        unet_vmfb = unet.export_unet_model(
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
            target=arguments["iree_target_triple"],
            ireec_flags=arguments["ireec_flags"],
            decomp_attn=arguments["decomp_attn"],
            attn_spec=arguments["attn_spec"],
            exit_on_vmfb=False,
        )
        arguments["external_weight_path"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_unet."
            + arguments["external_weights"]
        )
        arguments["vmfb_path"] = unet_vmfb
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
        atol = 4e-1

        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    def test03_ExportVaeModelDecode(self):
        if arguments["device"] in ["vulkan", "cuda"]:
            self.skipTest("Compilation error on vulkan; To be tested on cuda.")
        vae_vmfb = vae.export_vae_model(
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
            target=arguments["iree_target_triple"],
            ireec_flags=arguments["ireec_flags"],
            decomp_attn=True,
            attn_spec=arguments["attn_spec"],
            exit_on_vmfb=False,
        )
        arguments["external_weight_path"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_vae_decode."
            + arguments["external_weights"]
        )
        arguments["vmfb_path"] = vae_vmfb
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
        atol = 4e-1

        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    def test04_ExportVaeModelEncode(self):
        if arguments["device"] in ["cpu", "vulkan", "cuda", "rocm"]:
            self.skipTest(
                "Compilation error on cpu, vulkan and rocm; To be tested on cuda."
            )
        vae_vmfb = vae.export_vae_model(
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
            target=arguments["iree_target_triple"],
            ireec_flags=arguments["ireec_flags"],
            decomp_attn=True,
            exit_on_vmfb=True,
        )
        arguments["external_weight_path"] = (
            self.safe_model_name
            + "_"
            + arguments["precision"]
            + "_vae_encode."
            + arguments["external_weights"]
        )
        arguments["vmfb_path"] = vae_vmfb
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
        np.testing.assert_allclose(torch_output, turbine, rtol, atol)

    def test05_t2i_generate_images(self):
        if arguments["device"] in ["vulkan", "cuda"]:
            self.skipTest(
                "Have issues with submodels on vulkan, cuda; ROCM hangs on mi250 despite submodels working."
            )
        from turbine_models.custom_models.sd_inference.sd_pipeline import (
            SharkSDPipeline,
        )

        decomp_attn = {
            "text_encoder": False,
            "unet": False,
            "vae": True,
        }
        sd_pipe = SharkSDPipeline(
            arguments["hf_model_name"],
            arguments["height"],
            arguments["width"],
            arguments["batch_size"],
            arguments["max_length"],
            arguments["precision"],
            arguments["device"],
            arguments["iree_target_triple"],
            ireec_flags=None,  # ireec_flags
            attn_spec=arguments["attn_spec"],
            decomp_attn=decomp_attn,
            pipeline_dir="test_vmfbs",  # pipeline_dir
            external_weights_dir="test_weights",  # external_weights_dir
            external_weights=arguments["external_weights"],
            num_inference_steps=arguments["num_inference_steps"],
            cpu_scheduling=True,
            scheduler_id=arguments["scheduler_id"],
            shift=None,  # shift
            use_i8_punet=False,
        )
        sd_pipe.prepare_all()
        sd_pipe.load_map()
        output = sd_pipe.generate_images(
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

    @pytest.mark.skip(reason="Needs sdxl_quantized branch of IREE")
    def test06_t2i_generate_images_punet(self):
        if arguments["device"] in ["vulkan", "cuda", "rocm"]:
            self.skipTest(
                "Have issues with submodels on vulkan, cuda; ROCM hangs on mi250 despite submodels working."
            )
        from turbine_models.custom_models.sd_inference.sd_pipeline import (
            SharkSDPipeline,
        )

        decomp_attn = {
            "text_encoder": False,
            "unet": False,
            "vae": True,
        }
        sd_pipe = SharkSDPipeline(
            arguments["hf_model_name"],
            arguments["height"],
            arguments["width"],
            arguments["batch_size"],
            arguments["max_length"],
            arguments["precision"],
            arguments["device"],
            arguments["iree_target_triple"],
            ireec_flags=None,  # ireec_flags
            attn_spec=arguments["attn_spec"],
            decomp_attn=decomp_attn,
            pipeline_dir="test_vmfbs",  # pipeline_dir
            external_weights_dir="test_weights",  # external_weights_dir
            external_weights=arguments["external_weights"],
            num_inference_steps=arguments["num_inference_steps"],
            cpu_scheduling=True,
            scheduler_id=arguments["scheduler_id"],
            shift=None,  # shift
            use_i8_punet=True,
        )
        sd_pipe.prepare_all()
        sd_pipe.load_map()
        output = sd_pipe.generate_images(
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
