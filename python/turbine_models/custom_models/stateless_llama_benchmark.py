# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import re
import numpy as np

from transformers import AutoTokenizer
from iree import runtime as ireert
import turbine_models.custom_models.stateless_llama as llama

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token, required"
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="meta-llama/Llama-2-7b-chat-hf",
)
parser.add_argument("--external_weight_file", type=str, default="")
parser.add_argument("--vmfb_path", type=str, default="")
parser.add_argument(
    "--benchmark_steps",
    type=int,
    help="number of times second vicuna (run_forward) is run in benchmark (# of tokens to benchmark)",
    default=10,
)


def run_benchmark(args):
    config = ireert.Config("local-task")

    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    if args.vmfb_path:
        mod = ireert.VmModule.mmap(config.vm_instance, args.vmfb_path)
    elif os.path.exists(f"{safe_name}.vmfb"):
        mod = ireert.VmModule.mmap(config.vm_instance, f"{safe_name}.vmfb")
    else:
        sys.exit("no vmfb_path provided, required for run_benchmark")

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_name,
        use_fast=False,
        use_auth_token=args.hf_auth_token,
    )

    initial_input = tokenizer(llama.prompt, return_tensors="pt")
    example_input_id = initial_input.input_ids
    input = np.asarray(example_input_id, dtype=None, order="C")
    input = np.reshape(input, (1,) + (input.shape))
    llama.benchmark_steps = args.benchmark_steps

    if args.external_weight_file:
        weights = args.external_weight_file
        results = ireert.benchmark_module(
            mod, "run_all", input, parameters=f"model={weights}"
        )
    else:
        results = ireert.benchmark_module(mod, "run_all", input)

    for benchmark_result in results:
        print(
            f"benchmark_name: {benchmark_result.benchmark_name}, time: {benchmark_result.time}, cpu_time: {benchmark_result.cpu_time}, iterations: {benchmark_result.iterations}, user_counters: {benchmark_result.user_counters}"
        )


if __name__ == "__main__":
    args = parser.parse_args()
    run_benchmark(args)
