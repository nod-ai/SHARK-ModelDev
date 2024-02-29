# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import numpy as np
import os
import re
import sys

from transformers import AutoTokenizer
from iree import runtime as ireert
from turbine_models.utils.benchmark import benchmark_module
import turbine_models.custom_models.stateless_llama as llama


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
parser.add_argument("--benchmark_vmfb_path", type=str, default="")
parser.add_argument("--llama_vmfb_path", type=str, default="")
parser.add_argument(
    "--steps",
    type=int,
    default=10,
    help="number of times second vicuna is run (# of tokens to benchmark)",
)
parser.add_argument(
    "--run_forward_only_benchmark",
    action="store_true",
    help="do not include inititalization in benchmark for accurate tok/s",
)


def run_benchmark(args):
    config = ireert.Config("local-task")

    if args.external_weight_file:
        index = ireert.ParameterIndex()
        index.load(args.external_weight_file)

    if not args.benchmark_vmfb_path:
        sys.exit("no benchmark_vmfb_path provided, required for run_benchmark")
    benchmark_mod = ireert.VmModule.mmap(config.vm_instance, args.benchmark_vmfb_path)

    if not args.llama_vmfb_path:
        sys.exit("no llama_vmfb_path provided, required for run_benchmark")

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_name,
        use_fast=False,
        use_auth_token=args.hf_auth_token,
    )

    initial_input = tokenizer(llama.prompt, return_tensors="pt")
    example_input_id = initial_input.input_ids
    input = []
    temp = np.asarray(example_input_id, dtype=None, order="C")
    input.append(temp)
    input.append(np.array(args.steps))

    vmfbs = []
    vmfbs.append(args.llama_vmfb_path)
    vmfbs.append(args.benchmark_vmfb_path)

    if args.external_weight_file:
        results = benchmark_module(
            benchmark_mod,
            "run",
            vmfbs,
            input,
            parameters=f"model={args.external_weight_file}",
        )
    else:
        results = benchmark_module(benchmark_mod, "run", vmfbs, input)

    for benchmark_result in results:
        print(
            f"benchmark_name: {benchmark_result.benchmark_name}, time: {benchmark_result.time}, cpu_time: {benchmark_result.cpu_time}, iterations: {benchmark_result.iterations}, user_counters: {benchmark_result.user_counters}"
        )


def run_forward_benchmark(args):
    print("HERE")
    # Create the config for the IREE runtime environment
    config = ireert.Config("local-task")

    if not args.benchmark_vmfb_path:
        sys.exit("no benchmark_vmfb_path provided, required for run_benchmark")
    benchmark_mod = ireert.VmModule.mmap(config.vm_instance, args.benchmark_vmfb_path)

    # Load the external weight file if provided
    if args.external_weight_file:
        index = ireert.ParameterIndex()
        index.load(args.external_weight_file)

    # Ensure model name is in a safe format
    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)

    # Load the .vmfb model file
    if args.llama_vmfb_path:
        mod = ireert.VmModule.mmap(config.vm_instance, args.llama_vmfb_path)
    elif os.path.exists(f"{safe_name}.vmfb"):
        mod = ireert.VmModule.mmap(config.vm_instance, f"{safe_name}.vmfb")
    else:
        raise FileNotFoundError("No vmfb_path provided, required for run_vmfb")

    # Prepare the modules for the IREE runtime context
    vm_modules = [mod, ireert.create_hal_module(config.vm_instance, config.device)]

    # Include parameter module if external weight file is used
    if args.external_weight_file:
        param_module = ireert.create_io_parameters_module(
            config.vm_instance, index.create_provider(scope="model")
        )
        vm_modules.insert(0, param_module)

    # Create the system context with the given configuration and modules
    ctx = ireert.SystemContext(vm_modules=vm_modules, config=config)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_name, use_fast=False, use_auth_token=args.hf_auth_token
    )

    # Convert the prompt to input tensor
    initial_input = tokenizer(llama.prompt, return_tensors="pt")
    example_input_id = initial_input.input_ids
    device_inputs = [ireert.asdevicearray(config.device, example_input_id)]

    # Get the compiled module
    ModuleCompiled = ctx.modules.state_update
    init_val = ModuleCompiled["run_initialize"](*device_inputs)

    input = []
    temp = np.asarray(init_val, dtype=None, order="C")
    input.append(temp)
    input.append(np.array(args.steps))

    vmfbs = []
    vmfbs.append(args.llama_vmfb_path)
    vmfbs.append(args.benchmark_vmfb_path)

    if args.external_weight_file:
        results = benchmark_module(
            benchmark_mod,
            "run",
            vmfbs,
            input,
            parameters=f"model={args.external_weight_file}",
        )
    else:
        results = benchmark_module(benchmark_mod, "run", vmfbs, input)

    for benchmark_result in results:
        print(
            f"benchmark_name: {benchmark_result.benchmark_name}, time: {benchmark_result.time}, cpu_time: {benchmark_result.cpu_time}, iterations: {benchmark_result.iterations}, user_counters: {benchmark_result.user_counters}"
        )
        print(
            f"estimate: avg. {args.steps/(float(benchmark_result.time[:-3])/1000)} tok/s"
        )


# Python Benchmarking Support for multiple modules

DTYPE_TO_ABI_TYPE = {
    np.dtype(np.float32): "f32",
    np.dtype(np.int32): "i32",
    np.dtype(np.int64): "i64",
    np.dtype(np.float64): "f64",
    np.dtype(np.int16): "i16",
    np.dtype(np.int8): "i8",
    np.dtype(np.bool_): "i1",
}


if __name__ == "__main__":
    args = parser.parse_args()
    if args.run_forward_only_benchmark:
        run_forward_benchmark(args)
    else:
        run_benchmark(args)
