# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--benchmark_mlir_path", type=str, default="", help="Path to benchmark mlir module"
)
parser.add_argument(
    "--device", type=str, default="llvm-cpu", help="llvm-cpu, cuda, vulkan, rocm"
)
parser.add_argument(
    "--iree_target_triple",
    type=str,
    default="host",
    help="Specify vulkan target triple or rocm/cuda target device.",
)
parser.add_argument("--vulkan_max_allocation", type=str, default="4294967296")


def create_benchmark_vmfb(args):
    if not args.benchmark_mlir_path:
        sys.exit("no benchmark_vmfb_path provided, required for run_benchmark")

    flags = [
        "--iree-input-type=torch",
        "--mlir-print-debuginfo",
        "--mlir-print-op-on-diagnostic=false",
        "--iree-llvmcpu-target-cpu-features=host",
        "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
        "--iree-stream-resource-index-bits=64",
        "--iree-vm-target-index-bits=64",
        "--iree-opt-const-expr-hoisting=False",
    ]
    device = args.device
    if device == "cpu" or device == "llvm-cpu":
        flags.append("--iree-llvmcpu-enable-ukernels=all")
        device = "llvm-cpu"
    elif device == "vulkan":
        flags.extend(
            [
                "--iree-vulkan-target-triple=" + args.iree_target_triple,
                "--iree-stream-resource-max-allocation-size="
                + args.vulkan_max_allocation,
            ]
        )
    elif device == "rocm":
        flags.extend(
            [
                "--iree-rocm-target-chip=" + args.iree_target_triple,
                "--iree-rocm-link-bc=true",
                "--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode",
                "--iree-vm-bytecode-module-strip-source-map=true",
                "--iree-opt-strip-assertions=true",
                "--iree-vm-target-truncate-unsupported-floats",
            ]
        )
    elif device == "cuda":
        flags.extend(
            [
                "--iree-hal-cuda-llvm-target-arch=" + args.iree_target_triple,
                "--iree-vm-bytecode-module-strip-source-map=true",
                "--iree-vm-target-truncate-unsupported-floats",
            ]
        )
    else:
        print("Unknown device kind: ", device)

    import iree.compiler as ireec

    flatbuffer_blob = ireec.compile_file(
        input_file=f"{args.benchmark_mlir_path}",
        target_backends=[device],
        extra_args=flags,
    )
    with open(f"benchmark.vmfb", "wb+") as f:
        f.write(flatbuffer_blob)
    print("saved to benchmark.vmfb")
    exit()


if __name__ == "__main__":
    args = parser.parse_args()
    create_benchmark_vmfb(args)
