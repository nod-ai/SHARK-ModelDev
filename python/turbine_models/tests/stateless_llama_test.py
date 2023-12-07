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


def test_export(quantization: Literal["int4", None], precision: Literal["f16", "f32"]):
    llama.export_transformer_model(
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        hf_auth_token=None,
        compile_to="vmfb",
        external_weights="safetensors",
        # external_weight_file="medllama2_7b_f16_int4.safetensors", Do not export weights because this doesn't get quantized
        quantization=quantization,
        precision=precision,
        device="llvm-cpu",
        target_triple="host",
    )

    from turbine_models.gen_external_params.gen_external_params import (
        gen_external_params,
    )

    gen_external_params(
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        quantization=quantization,
        weight_path="medllama2_7b_f16_int4.safetensors",
        hf_auth_token=None,
        precision=precision,
    )


    # def run_vmfb_comparison(prompt, hf_auth_token, hf_model_name, vmfb_path, external_weight_file, break_on_eos=True):
    DEFAULT_PROMPT = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
"""

    from .vmfb_comparison import run_vmfb_comparison
    turbine_str, torch_str = run_vmfb_comparison(
        prompt=DEFAULT_PROMPT,
        hf_auth_token=None,
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        vmfb_path="medllama2_7b.vmfb",
        external_weight_file="medllama2_7b_f16_int4.safetensors",
        break_on_eos=True,
    )

    import difflib
        # Calculate and print diff
    diff = difflib.unified_diff(
        torch_str.splitlines(keepends=True),
        turbine_str.splitlines(keepends=True),
        fromfile='torch_str',
        tofile='turbine_str',
        lineterm=''
    )
    
    assert torch_str == turbine_str, "".join(diff)

    



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
