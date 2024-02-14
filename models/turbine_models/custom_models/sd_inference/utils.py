import iree.compiler as ireec
import numpy as np
import safetensors
import re


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


def compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name):
    flags = [
        "--iree-input-type=torch",
        "--mlir-print-debuginfo",
        "--mlir-print-op-on-diagnostic=false",
        "--iree-llvmcpu-target-cpu-features=host",
        "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
        "--iree-stream-resource-index-bits=64",
        "--iree-vm-target-index-bits=64",
    ]
    if device == "cpu":
        flags.append("--iree-llvmcpu-enable-ukernels=all")
        device = "llvm-cpu"
    elif device == "vulkan":
        flags.extend(
            [
                "--iree-hal-target-backends=vulkan-spirv",
                "--iree-vulkan-target-triple=" + target_triple,
                "--iree-stream-resource-max-allocation-size=" + max_alloc,
            ]
        )
    elif device == "rocm":
        flags.extend(
            [
                "--iree-hal-target-backends=rocm",
                "--iree-rocm-target-chip=" + target_triple,
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
                "--iree-hal-target-backends=cuda",
                "--iree-hal-cuda-llvm-target-arch=" + target_triple,
                "--iree-vm-bytecode-module-strip-source-map=true",
                "--iree-vm-target-truncate-unsupported-floats",
            ]
        )
    else:
        print("incorrect device: ", device)

    flatbuffer_blob = ireec.compile_str(
        module_str,
        target_backends=[device],
        extra_args=flags,
    )
    with open(f"{safe_name}.vmfb", "wb+") as f:
        f.write(flatbuffer_blob)
    print("Saved to", safe_name + ".vmfb")
    return


def create_safe_name(hf_model_name, model_name_str):
    safe_name = hf_model_name.split("/")[-1].strip() + model_name_str
    safe_name = re.sub("-", "_", safe_name)
    return safe_name
