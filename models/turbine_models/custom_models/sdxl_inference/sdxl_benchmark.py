# Copyright 2024 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numpy as np
import torch
import sys

from iree import runtime as ireert
from turbine_models.utils.benchmark import benchmark_module


def run_benchmark(args):
    config = ireert.Config(args.rt_device)

    if args.external_weight_file:
        index = ireert.ParameterIndex()
        index.load(args.external_weight_file)

    if not args.benchmark_vmfb_path:
        sys.exit("no --benchmark_vmfb_path provided, required for run_benchmark")
    benchmark_mod = ireert.VmModule.mmap(config.vm_instance, args.benchmark_vmfb_path)

    if not args.scheduled_unet_vmfb_path:
        sys.exit("no --scheduled_unet_vmfb_path provided, required for run_benchmark")

    dtype = np.float16 if args.precision == "fp16" else np.float32
    sample = np.random.randn(
        args.batch_size, 4, args.height // 8, args.width // 8
    ).astype(dtype)
    prompt_embeds = np.random.randn(2 * args.batch_size, args.max_length, 2048).astype(
        dtype
    )
    text_embeds = np.random.randn(2 * args.batch_size, 1280).astype(dtype)
    guidance_scale = np.array([7.5], dtype=dtype)
    num_iters = np.array(args.num_inference_steps)
    input = [
        sample,
        prompt_embeds,
        text_embeds,
        guidance_scale,
        num_iters,
    ]

    vmfbs = []
    vmfbs.append(args.scheduled_unet_vmfb_path)
    vmfbs.append(args.benchmark_vmfb_path)

    if args.external_weight_file:
        results = benchmark_module(
            benchmark_mod,
            "produce_image_latents",
            vmfbs,
            input,
            parameters=f"model={args.external_weight_file}",
        )
    else:
        results = benchmark_module(benchmark_mod, "produce_image_latents", vmfbs, input)
    for benchmark_result in results:
        print(
            f"benchmark_name: {benchmark_result.benchmark_name}, time: {benchmark_result.time}, cpu_time: {benchmark_result.cpu_time}, iterations: {benchmark_result.iterations}, user_counters: {benchmark_result.user_counters}"
        )


# Python Benchmarking Support for multiple modules

if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    run_benchmark(args)
