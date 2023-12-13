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


def create_benchmark_vmfb(args):
    if not args.benchmark_mlir_path:
        sys.exit("no benchmark_vmfb_path provided, required for run_benchmark")

    flags = [
        "--iree-input-type=torch",
        "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
        "--mlir-print-debuginfo",
        "--mlir-print-op-on-diagnostic=false",
        "--iree-llvmcpu-target-cpu-features=host",
        "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
        "--iree-llvmcpu-enable-microkernels",
        "--iree-llvmcpu-stack-allocation-limit=256000",
        "--iree-stream-resource-index-bits=64",
        "--iree-vm-target-index-bits=64",
        "--iree-vm-bytecode-module-strip-source-map=true",
        "--iree-util-zero-fill-elided-attrs",
        "--iree-vm-target-truncate-unsupported-floats",
        "--iree-codegen-check-ir-before-llvm-conversion=false",
        "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
        "--iree-opt-const-expr-hoisting=False",
    ]

    import iree.compiler as ireec

    flatbuffer_blob = ireec.compile_file(
        input_file=f"{args.benchmark_mlir_path}",
        target_backends=["llvm-cpu"],
        extra_args=flags,
    )
    with open(f"benchmark.vmfb", "wb+") as f:
        f.write(flatbuffer_blob)
    print("saved to benchmark.vmfb")
    exit()


if __name__ == "__main__":
    args = parser.parse_args()
    create_benchmark_vmfb(args)
