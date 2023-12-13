# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import numpy as np

from transformers import AutoTokenizer
from iree import runtime as ireert
import turbine_models.custom_models.stateless_llama as llama

import argparse

import subprocess
from collections import namedtuple

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

    if args.external_weight_file:
        results = benchmark_module(
            benchmark_mod,
            args,
            "run",
            input,
            parameters=f"model={args.external_weight_file}",
        )
    else:
        results = benchmark_module(benchmark_mod, args, "run", input)

    for benchmark_result in results:
        print(
            f"benchmark_name: {benchmark_result.benchmark_name}, time: {benchmark_result.time}, cpu_time: {benchmark_result.cpu_time}, iterations: {benchmark_result.iterations}, user_counters: {benchmark_result.user_counters}"
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

BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)


class BenchmarkToolError(Exception):
    """Benchmark exception that preserves the command line and error output."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class BenchmarkTimeoutError(Exception):
    """Exception raised if the benchmark is cancelled by the user specified timeout."""

    pass


def benchmark_module(
    module, bench_args, entry_function=None, inputs=[], timeout=None, **kwargs
):
    funcs = [a for a in module.function_names if a != "__init"]
    if entry_function is None:
        if len(funcs) > 1:
            raise ValueError(f"No function specified with multiple options {funcs}")
        entry_function = funcs[0]
    if entry_function not in funcs:
        raise ValueError(
            f"Attempted to benchmark unknown function {entry_function} of options {funcs}"
        )

    args = [ireert.benchmark_exe()]
    args.append(f"--function={entry_function}")

    for inp in inputs:
        if isinstance(inp, str):
            args.append(f"--input={inp}")
            continue
        shape = "x".join([str(d) for d in inp.shape])
        abitype = DTYPE_TO_ABI_TYPE[inp.dtype]
        values = inp.flatten()
        if np.all(values[0] == values):
            values = str(values[0])
        else:
            values = ",".join([str(v) for v in values])

        args.append(f"--input={shape}x{abitype}={values}")

    for k in kwargs:
        v = kwargs[k]
        args.append(f"--{k}={v}")

    args.append(f"--module={bench_args.llama_vmfb_path}")
    args.append(f"--module={bench_args.benchmark_vmfb_path}")

    try:
        benchmark_process = subprocess.run(
            args=args,
            # input=flatbuffer,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.TimeoutExpired:
        raise BenchmarkTimeoutError(f"Benchmark timed out after {timeout} seconds")
    out = benchmark_process.stdout
    err = benchmark_process.stderr

    err = err.decode()
    if "INVALID_ARGUMENT;" in err:
        raise ValueError("Invalid inputs specified for benchmarking")

    # In the event benchmarking runs but encounteres an internal error,
    # return the internal error instead of benchmark results.
    if "INTERNAL; CUDA driver error" in str(out):
        raise BenchmarkToolError(str(out))

    # Grab individual results by line (skip header lines)
    bench_lines = out.decode().split("\n")[3:]
    benchmark_results = []
    for line in bench_lines:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark_name = split[0]
        time = " ".join(split[1:3])
        cpu_time = " ".join(split[3:5])
        iterations = split[5]
        user_counters = None
        if len(split) > 5:
            user_counters = split[6]
        benchmark_results.append(
            BenchmarkResult(
                benchmark_name=benchmark_name,
                time=time,
                cpu_time=cpu_time,
                iterations=iterations,
                user_counters=user_counters,
            )
        )

    return benchmark_results


if __name__ == "__main__":
    args = parser.parse_args()
    run_benchmark(args)
