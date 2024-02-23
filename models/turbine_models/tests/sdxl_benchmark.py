import subprocess
import sys
from collections import namedtuple
from iree import runtime as ireert
from turbine_models.custom_models.llama_benchmark.stateless_llama_benchmark import (
    benchmark_module,
)


DTYPE_MAP = {
    "fp16": "f16",
    "fp32": "f32",
}


def run_benchmark(
    model,
    vmfb_path,
    weights_path,
    device,
    max_length=None,
    height=None,
    width=None,
    batch_size=None,
    in_channels=None,
    precision=None,
):
    config = ireert.Config(device)

    if not vmfb_path:
        sys.exit("no vmfb_path provided, required for run_benchmark")
    benchmark_mod = ireert.VmModule.mmap(config.vm_instance, vmfb_path)

    if weights_path:
        index = ireert.ParameterIndex()
        index.load(weights_path)

    vmfbs = []
    vmfbs.append(vmfb_path)

    inputs = []
    match model:
        case "clip_1":
            inputs.append(f"1x{max_length}xi64")
        case "clip_2":
            inputs.append(f"1x{max_length}xi64")
        case "unet":
            inputs.append(
                f"{batch_size}x{in_channels}x{height//8}x{width//8}x{DTYPE_MAP[precision]}"
            )
            inputs.append(f"1x{DTYPE_MAP[precision]}")
            inputs.append(f"{2*batch_size}x{max_length}x2048x{DTYPE_MAP[precision]}")
            inputs.append(f"{2*batch_size}x1280x{DTYPE_MAP[precision]}")
            inputs.append(f"{2*batch_size}x6x{DTYPE_MAP[precision]}")
            inputs.append(f"1x{DTYPE_MAP[precision]}")
        case "vae_decode":
            inputs.append(f"1x4x{height//8}x{width//8}x{DTYPE_MAP[precision]}")
        case "vae_encode":
            inputs.append(f"1x3x{height}x{width}x{DTYPE_MAP[precision]}")
        case _:
            sys.exit("model name doesn't match for inputs")

    if weights_path:
        results = benchmark_module(
            benchmark_mod,
            "main",
            vmfbs,
            inputs,
            parameters=f"model={weights_path}",
        )
    else:
        results = benchmark_module(benchmark_mod, "main", vmfbs, inputs)

    for benchmark_result in results:
        print(
            f"model: {model}, benchmark_name: {benchmark_result.benchmark_name}, time: {benchmark_result.time}, cpu_time: {benchmark_result.cpu_time}, iterations: {benchmark_result.iterations}, user_counters: {benchmark_result.user_counters}"
        )
