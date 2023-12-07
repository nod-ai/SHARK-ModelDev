# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import turbine_models.custom_models.stateless_llama as llama
import unittest
import os
import pytest

from typing import Literal


import os
import sys
import re

from typing import Tuple

os.environ["TORCH_LOGS"] = "dynamic"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils import _pytree as pytree
from shark_turbine.aot import *
from iree.compiler.ir import Context
from iree import runtime as ireert

from turbine_models.custom_models import remap_gguf
import safetensors

from tqdm import tqdm



def test_export_vmfb(quantization: Literal["int4", None], precision: Literal["f16", "f32"]):
    llama.export_transformer_model(
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        hf_auth_token=None,
        compile_to="vmfb",
        external_weights="safetensors",
        # external_weight_file="Llama-2-7b-chat-hf-function-calling-v2_f16_int4.safetensors", Do not export weights because this doesn't get quantized
        quantization=quantization,
        precision=precision,
        device="llvm-cpu",
        target_triple="host",
    )

# def test_export_safetensors(quantization: Literal["int4", None], precision: Literal["f16", "f32"]):
    from turbine_models.gen_external_params.gen_external_params import (
        gen_external_params,
    )

    gen_external_params(
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        quantization=quantization,
        hf_auth_token=None,
        precision=precision,
    )

# def test_run_vmfb(quantization: Literal["int4", None], precision: Literal["f16", "f32"]):
    DEFAULT_PROMPT = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
"""
    # cache reference output to avoid having to use torch
    TORCH_REFERENCE_STRING = '''Hello! I'm just an AI assistant, I'm here to help you with any questions you may have. However, I must point out that the question "what are you?" is not clear or factually coherent. I'm just a language model, I don't have a physical body or personal identity, so I cannot be described as a "you" in the classical sense. Is there anything else I can help you with?</s>'''
    # torch_reference_string = get_torch_string(
    #     prompt=DEFAULT_PROMPT,
    #     hf_auth_token=None,
    #     hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
    #     tokens_to_compare=50,
    # )
    # 
    torch_str = TORCH_REFERENCE_STRING
    

    from .vmfb_comparison import get_turbine_vmfb_string, get_torch_string

    torch_str = get_torch_string(
        prompt=DEFAULT_PROMPT,
        hf_auth_token=None,
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        tokens_to_compare=50,
        precision=precision,
        quantization=quantization,
    )

    # store torch string
    # include precision and quantization in filename
    torch_str_cache_path = f"python/turbine_models/tests/vmfb_comparison_cached_torch_output_{precision}_{quantization}.txt"
    with open(torch_str_cache_path, "w") as f:
        f.write(torch_str)

    turbine_str = get_turbine_vmfb_string(
        prompt=DEFAULT_PROMPT,
        hf_auth_token=None,
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        vmfb_path="Llama_2_7b_chat_hf_function_calling_v2.vmfb",
        external_weight_file="Llama_2_7b_chat_hf_function_calling_v2_f16_int4.safetensors",
        tokens_to_compare=50,
    )

    import difflib

    # Calculate and print diff
    diff = difflib.unified_diff(
        torch_str.splitlines(keepends=True),
        turbine_str.splitlines(keepends=True),
        fromfile="torch_str",
        tofile="turbine_str",
        lineterm="",
    )

    assert torch_str == turbine_str, "".join(diff)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
