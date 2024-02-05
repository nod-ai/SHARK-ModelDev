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
import json

os.environ["TORCH_LOGS"] = "dynamic"
from shark_turbine.aot import *
from turbine_models.custom_models import llm_runner
from turbine_models.custom_models.llama_benchmark.e2e import llm_e2e_benchmark
from turbine_models.gen_external_params.gen_external_params import (
    gen_external_params,
)

quantization = "unquantized"
precision = "f32"
gen_external_params(
    hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
    quantization=quantization,
    hf_auth_token=None,
    precision=precision,
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
    assert reference == output, "".join(diff)


class StatelessLlamaChecks(unittest.TestCase):
    def test_vmfb_comparison(self):
        """
        Test that the vmfb model produces the same output as the torch model

        Precision can be 16 or 32, using 16 for speed and memory.

        For VMFB, quantization can be int4 or None, but right now only using none for compatibility with torch.
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
        )

        torch_str_cache_path = f"python/turbine_models/tests/vmfb_comparison_cached_torch_output_{precision}_{quantization}.txt"
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
            f"Llama_2_7b_chat_hf_function_calling_v2_{precision}_{quantization}.safetensors",
        )
        check_output_string(torch_str, turbine_str)

    def test_benchmark_vmfb(self):
        vmfb_name = "Llama_2_7b_chat_hf_function_calling_v2.vmfb"
        if not os.path.isfile(vmfb_name):
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
        test_dataset_path = "python/turbine_models/tests/benchmark_prompt_test.json"
        test_output_path = "benchmark_e2e_results.json"
        benchmark_result_path = llm_e2e_benchmark.run_llm_benchmark(
            "local-task",
            "Llama_2_7b_chat_hf_function_calling_v2.vmfb",
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            None,
            f"Llama_2_7b_chat_hf_function_calling_v2_{precision}_{quantization}.safetensors",
            test_dataset_path,
            test_output_path,
        )
        benchmark_result = []
        with open(benchmark_result_path) as f:
            benchmark_result = json.load(f)
        if len(benchmark_result) <= 0:
            raise ValueError("Dataset is empty, or did not read dataset correctly.")
        # Test result for prompt #1
        assert benchmark_result[0]["decoded_tokens"] == 10
        assert benchmark_result[0]["num_iterations"] == 2
        assert benchmark_result[0]["decode_speed(tok/s)"] > 0
        assert benchmark_result[0]["prefill_speed(tok/s)"] > 0
        # Test result for prompt #2
        assert benchmark_result[1]["decoded_tokens"] == 25
        assert benchmark_result[1]["num_iterations"] == 1
        assert benchmark_result[1]["decode_speed(tok/s)"] > 0
        assert benchmark_result[1]["prefill_speed(tok/s)"] > 0

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
        )

        torch_str_cache_path = f"python/turbine_models/tests/vmfb_comparison_cached_torch_output_{precision}_{quantization}.txt"
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
        )
        check_output_string(torch_str, turbine_str)

    def test_rerotated_torch_comparison(self):
        torch_str = llm_runner.run_torch_llm(
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            None,
            DEFAULT_PROMPT,
        )
        rotated_torch_str = llm_runner.run_torch_llm(
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            None,
            DEFAULT_PROMPT,
            streaming_llm=True,
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
