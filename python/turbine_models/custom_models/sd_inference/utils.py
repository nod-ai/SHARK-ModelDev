import iree.compiler as ireec
import numpy as np
import safetensors


def save_external_weights(
    mapper,
    model,
    external_weights=None,
    external_weight_file=None,
):
    if external_weights is not None:
        if external_weights == "safetensors":
            mod_params = dict(model.named_parameters())
            for name in mod_params:
                mapper["params." + name] = name
            if external_weight_file:
                safetensors.torch.save_file(mod_params, external_weight_file)
                print("Saved params to", external_weight_file)


def largest_error(array1, array2):
    absolute_diff = np.abs(array1 - array2)
    max_error = np.max(absolute_diff)
    return max_error


def compile_to_vmfb(module_str, target_backends, safe_name):
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

    flatbuffer_blob = ireec.compile_str(
        module_str,
        target_backends=target_backends,
        extra_args=flags,
    )
    with open(f"{safe_name}.vmfb", "wb+") as f:
        f.write(flatbuffer_blob)
    print("Saved to", safe_name + ".vmfb")
    exit()
