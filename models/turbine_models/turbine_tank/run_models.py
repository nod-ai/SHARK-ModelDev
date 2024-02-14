# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from turbine_models.custom_models.sd_inference import (
    clip,
    clip_runner,
    unet,
    unet_runner,
    vae,
    vae_runner,
)

from turbine_models.custom_models.sd_inference import utils
import torch
import os
import turbine_models.custom_models.stateless_llama as llama
import difflib
from turbine_models.turbine_tank import turbine_tank

parser = argparse.ArgumentParser()
parser.add_argument(
    "--download_ir",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="download IR from turbine tank",
)
parser.add_argument(
    "--upload_ir",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="upload IR to turbine tank",
)

os.environ["TORCH_LOGS"] = "dynamic"
from shark_turbine.aot import *
from turbine_models.custom_models import llm_runner

from turbine_models.gen_external_params.gen_external_params import (
    gen_external_params,
)

DEFAULT_PROMPT = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
"""


def check_output_string(reference, output):
    # Calculate and print diff
    diff = difflib.unified_diff(
        reference.splitlines(keepends=True),
        output.splitlines(keepends=True),
        fromfile="reference",
        tofile="output",
        lineterm="",
    )
    return "".join(diff)


def run_llama_model(download_ir=False, upload_ir=True):
    if not download_ir:
        gen_external_params(
            hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            hf_auth_token=None,
        )
    llama.export_transformer_model(
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        hf_auth_token=None,
        compile_to="vmfb",
        external_weights="safetensors",
        # external_weight_file="Llama-2-7b-chat-hf-function-calling-v2_f16_int4.safetensors", Do not export weights because this doesn't get quantized
        quantization="int4",
        precision="f16",
        device="llvm-cpu",
        target_triple="host",
        download_ir=download_ir,
        upload_ir=upload_ir,
    )

    if download_ir:
        return

    torch_str_cache_path = (
        f"models/turbine_models/tests/vmfb_comparison_cached_torch_output_f16_int4.txt"
    )
    # if cached, just read
    if os.path.exists(torch_str_cache_path):
        with open(torch_str_cache_path, "r") as f:
            torch_str = f.read()
    else:
        torch_str = llm_runner.run_torch_llm(
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2", None, DEFAULT_PROMPT
        )

        with open(torch_str_cache_path, "w") as f:
            f.write(torch_str)

    turbine_str = llm_runner.run_llm(
        "local-task",
        DEFAULT_PROMPT,
        "Llama_2_7b_chat_hf_function_calling_v2.vmfb",
        "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        None,
        f"Llama_2_7b_chat_hf_function_calling_v2_f16_int4.safetensors",
    )

    result = check_output_string(torch_str, turbine_str)

    # clean up
    os.remove("Llama_2_7b_chat_hf_function_calling_v2_f16_int4.safetensors")
    os.remove("Llama_2_7b_chat_hf_function_calling_v2.vmfb")
    os.remove("Llama_2_7b_chat_hf_function_calling_v2.mlir")

    return result


arguments = {
    "hf_auth_token": None,
    "hf_model_name": "CompVis/stable-diffusion-v1-4",
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


def run_clip_model(download_ir=False, upload_ir=True):
    clip.export_clip_model(
        # This is a public model, so no auth required
        "CompVis/stable-diffusion-v1-4",
        None,
        "vmfb",
        "safetensors",
        "stable_diffusion_v1_4_clip.safetensors",
        "cpu",
        download_ir=download_ir,
        upload_ir=upload_ir,
    )

    if download_ir:
        return

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
    if err < 9e-5:
        result = "CLIP SUCCESS: " + str(err)
    else:
        result = "CLIP FAILURE: " + str(err)

    # clean up
    os.remove("stable_diffusion_v1_4_clip.safetensors")
    os.remove("stable_diffusion_v1_4_clip.vmfb")
    os.remove("stable_diffusion_v1_4_clip.mlir")

    return result


def run_unet_model(download_ir=False, upload_ir=True):
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
        download_ir=download_ir,
        upload_ir=upload_ir,
    )

    if download_ir:
        return

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
    if err < 9e-5:
        result = "UNET SUCCESS: " + str(err)
    else:
        result = "UNET FAILURE: " + str(err)

    # clean up
    os.remove("stable_diffusion_v1_4_unet.safetensors")
    os.remove("stable_diffusion_v1_4_unet.vmfb")
    os.remove("stable_diffusion_v1_4_unet.mlir")

    return result


def run_vae_decode(download_ir=False, upload_ir=True):
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
        download_ir=download_ir,
        upload_ir=upload_ir,
    )

    if download_ir:
        return

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
    if err < 9e-5:
        result = "VAE DECODE SUCCESS: " + str(err)
    else:
        result = "VAE DECODE FAILURE: " + str(err)

    # clean up
    os.remove("stable_diffusion_v1_4_vae.safetensors")
    os.remove("stable_diffusion_v1_4_vae.vmfb")
    os.remove("stable_diffusion_v1_4_vae.mlir")

    return result


def run_vae_encode(download_ir=False, upload_ir=True):
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
        download_ir=download_ir,
        upload_ir=upload_ir,
    )

    if download_ir:
        return

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
    if err < 2e-3:
        result = "VAE ENCODE SUCCESS: " + str(err)
    else:
        result = "VAE ENCODE FAILURE: " + str(err)

    # clean up
    os.remove("stable_diffusion_v1_4_vae.safetensors")
    os.remove("stable_diffusion_v1_4_vae.vmfb")
    os.remove("stable_diffusion_v1_4_vae.mlir")

    return result


if __name__ == "__main__":
    args = parser.parse_args()

    if args.upload_ir and args.download_ir:
        raise ValueError("upload_ir and download_ir can't both be true")

    if args.upload_ir:
        result = "Turbine Tank Results\n"
        llama_result = run_llama_model(args.download_ir, args.upload_ir)
        result += llama_result + "\n"
        clip_result = run_clip_model(args.download_ir, args.upload_ir)
        result += clip_result + "\n"
        unet_result = run_unet_model(args.download_ir, args.upload_ir)
        result += unet_result + "\n"
        vae_decode_result = run_vae_decode(args.download_ir, args.upload_ir)
        result += vae_decode_result + "\n"
        vae_encode_result = run_vae_encode(args.download_ir, args.upload_ir)
        result += vae_encode_result + "\n"
        f = open("daily_report.txt", "a")
        f.write(result)
        f.close()
        turbine_tank.uploadToBlobStorage(
            str(os.path.abspath("daily_report.txt")), "daily_report.txt"
        )
        os.remove("daily_report.txt")
    else:
        run_llama_model(args.download_ir, args.upload_ir)
        run_clip_model(args.download_ir, args.upload_ir)
        run_unet_model(args.download_ir, args.upload_ir)
        run_vae_decode(args.download_ir, args.upload_ir)
        run_vae_encode(args.download_ir, args.upload_ir)
