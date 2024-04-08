# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import turbine_models.custom_models.stateless_llama as llama
import os
import unittest
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from accelerate import init_empty_weights
from transformers.modeling_utils import load_sharded_checkpoint
import tempfile

os.environ["TORCH_LOGS"] = "dynamic"
from shark_turbine.aot import *
from turbine_models.custom_models import llm_runner

from turbine_models.gen_external_params.gen_external_params import (
    gen_external_params,
)
from turbine_models.turbine_tank import turbine_tank


def check_output_string(reference, output):
    # Calculate and print diff
    diff = difflib.unified_diff(
        reference.splitlines(keepends=True),
        output.splitlines(keepends=True),
        fromfile="reference",
        tofile="output",
        lineterm="",
    )
    assert reference == output, "".join(diff)


quantization = "unquantized"
precision = "f32"

DEFAULT_PROMPT = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
"""


class StatelessLlamaChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gen_external_params(
            hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            quantization=quantization,
            hf_auth_token=None,
            precision=precision,
        )

        cls.tokenizer = AutoTokenizer.from_pretrained(
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            use_fast=False,
        )

        # The model is first created on the Meta device (with empty weights) and the state dict
        # is then loaded inside it (shard by shard in the case of a sharded checkpoint).
        # This avoids using twice the size of model with creating whole model with random weights,
        # then loading pretrained weights.
        cls.mod = AutoModelForCausalLM.from_pretrained(
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            torch_dtype=torch.float,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    @classmethod
    def tearDownClass(cls):
        cls.tokenizer = None
        cls.mod = None

    # See: https://github.com/nod-ai/SHARK-Turbine/issues/560
    # Developed issues related to the pytorch 2.3 upgrade.
    # @unittest.expectedFailure
    def test_vmfb_comparison(self):
        """
        Test that the vmfb model produces the same output as the torch model

        Precision can be 16 or 32, using 16 for speed and memory.

        For VMFB, quantization can be int4 or None, but right now only using none for compatibility with torch.
        """

        upload_ir_var = os.environ.get("TURBINE_TANK_ACTION", "not_upload")

        blob_name = llama.export_transformer_model(
            hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            hf_auth_token=None,
            compile_to="vmfb",
            external_weights="safetensors",
            # external_weight_file="Llama-2-7b-chat-hf-function-calling-v2_f16_int4.safetensors", Do not export weights because this doesn't get quantized
            quantization=quantization,
            precision=precision,
            device="llvm-cpu",
            target_triple="host",
            upload_ir=upload_ir_var == "upload",
            mod=self.mod,
            tokenizer=self.tokenizer,
        )

        torch_str_cache_path = (
            f"vmfb_comparison_cached_torch_output_{precision}_{quantization}.txt"
        )
        # if cached, just read
        if os.path.exists(torch_str_cache_path):
            with open(torch_str_cache_path, "r") as f:
                torch_str = f.read()
        else:
            torch_str = llm_runner.run_torch_llm(
                "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
                None,
                DEFAULT_PROMPT,
                model=self.mod,
                tokenizer=self.tokenizer,
            )

            with open(torch_str_cache_path, "w") as f:
                f.write(torch_str)

        turbine_str = llm_runner.run_llm(
            "local-task",
            DEFAULT_PROMPT,
            "Llama_2_7b_chat_hf_function_calling_v2.vmfb",
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            None,
            f"Llama_2_7b_chat_hf_function_calling_v2_{precision}_{quantization}.safetensors",
            tokenizer=self.tokenizer,
        )
        check_output_string(torch_str, turbine_str)
        if upload_ir_var == "upload":
            new_blob_name = blob_name.split(".")
            new_blob_name = new_blob_name[0] + "-pass.mlir"
            turbine_tank.changeBlobName(blob_name, new_blob_name)

    def test_streaming_vmfb_comparison(self):
        """
        Similar test to above but for streaming-LLM.
        """
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
            streaming_llm=True,
            vmfb_path="streaming_llama.vmfb",
            mod=None,
            tokenizer=self.tokenizer,
        )

        torch_str_cache_path = (
            f"vmfb_comparison_cached_torch_output_{precision}_{quantization}.txt"
        )
        # if cached, just read
        if os.path.exists(torch_str_cache_path):
            with open(torch_str_cache_path, "r") as f:
                torch_str = f.read()
        else:
            torch_str = llm_runner.run_torch_llm(
                "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
                None,
                DEFAULT_PROMPT,
                streaming_llm=True,
                model=None,
                tokenizer=self.tokenizer,
            )

            with open(torch_str_cache_path, "w") as f:
                f.write(torch_str)

        turbine_str = llm_runner.run_llm(
            "local-task",
            DEFAULT_PROMPT,
            "streaming_llama.vmfb",
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            None,
            f"Llama_2_7b_chat_hf_function_calling_v2_{precision}_{quantization}.safetensors",
            streaming_llm=True,
            tokenizer=self.tokenizer,
        )
        check_output_string(torch_str, turbine_str)

    # See: https://github.com/nod-ai/SHARK-Turbine/issues/560
    # Developed issues related to the pytorch 2.3 upgrade.
    @unittest.expectedFailure
    def test_rerotated_torch_comparison(self):
        torch_str = llm_runner.run_torch_llm(
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            None,
            DEFAULT_PROMPT,
            model=self.mod,
            tokenizer=self.tokenizer,
        )
        rotated_torch_str = llm_runner.run_torch_llm(
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            None,
            DEFAULT_PROMPT,
            streaming_llm=True,
            model=None,
            tokenizer=self.tokenizer,
        )
        check_output_string(torch_str, rotated_torch_str)

    def test_kvcachce_schema(self):
        json_schema_16 = """[1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}]}]"""
        num_layers = 8
        auto_schema_16 = llama.generate_schema(num_layers)
        assert auto_schema_16 == json_schema_16

        json_schema_64 = """[1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}]}]"""
        num_layers = 32
        auto_schema_64 = llama.generate_schema(num_layers)
        assert auto_schema_64 == json_schema_64


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
