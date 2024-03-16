import iree.compiler as ireec
import numpy as np
import os
import safetensors
import re
from diffusers import (
    PNDMScheduler,
)


# If flags are verified to work on a specific model and improve performance without regressing numerics, add them to this dictionary. If you are working with bleeding edge flags, please add them manually with the --ireec_flags argument.
gfx94X_flags = {
    "all": [
        "--iree-global-opt-propagate-transposes=true",
        "--iree-opt-const-eval=false",
        "--iree-opt-outer-dim-concat=true",
        "--iree-codegen-gpu-native-math-precision=true",
        "--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode",
        "--iree-vm-target-truncate-unsupported-floats",
        "--iree-llvmgpu-enable-prefetch=true",
        "--verify=false",
        "--iree-codegen-log-swizzle-tile=4",
        "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics)",
    ],
    "unet": [
        "--iree-codegen-llvmgpu-use-vector-distribution",
    ],
    "clip": [
        "--iree-flow-split-matmul-reduction=1",
        "--iree-global-opt-only-sink-transposes=true",
    ],
    "vae": [
        "--iree-codegen-llvmgpu-use-vector-distribution",
        "--iree-global-opt-only-sink-transposes=true",
    ],
}


def compile_to_vmfb(
    module_str,
    device,
    target_triple,
    ireec_flags,
    safe_name,
    return_path=False,
    const_expr_hoisting=True,
    mlir_source="str",
    max_alloc="4294967296",
    save_mlir=False,
    attn_spec=None,
):
    flags = []
    if target_triple in ["", None] and "triple" not in ireec_flags:
        raise ValueError(
            "target_triple must be set. Usually this can be fixed by setting --iree_target_triple in the CLI."
        )
    if device == "cpu":
        flags.extend(
            [
                "--iree-llvmcpu-target-triple=" + target_triple,
                "--iree-llvmcpu-target-cpu-features=host",
                "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false",
                "--iree-llvmcpu-distribution-size=32",
            ]
        )
        device = "llvm-cpu"
    elif device == "vulkan":
        flags.extend(
            [
                "--iree-hal-target-backends=vulkan-spirv",
                "--iree-vulkan-target-triple=" + target_triple,
                "--iree-stream-resource-max-allocation-size=" + max_alloc,
                "--iree-stream-resource-index-bits=64",
                "--iree-vm-target-index-bits=64",
                "--iree-flow-inline-constants-max-byte-length=1",
            ]
        )
        device = "vulkan-spirv"
    elif device == "rocm":
        flags.extend(
            [
                "--iree-hal-target-backends=rocm",
                "--iree-rocm-target-chip=" + target_triple,
                "--iree-rocm-link-bc=true",
                "--verify=false",
            ]
        )
    elif device == "cuda":
        flags.extend(
            [
                "--iree-hal-target-backends=cuda",
                "--iree-hal-cuda-llvm-target-arch=" + target_triple,
                "--iree-vm-target-truncate-unsupported-floats",
            ]
        )
    else:
        print("incorrect device: ", device)
    if const_expr_hoisting == False:
        flags.extend(
            [
                "--iree-opt-const-expr-hoisting=False",
                "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
            ]
        )
    if isinstance(ireec_flags, str):
        if ireec_flags != "":
            ireec_flags = ireec_flags.split(",")

    for i, flag in enumerate(ireec_flags):
        k = flag.strip().split("=")[0]
        for idx, default in enumerate(flags):
            if k == default.split("=")[0]:
                flags[idx] = flag
                ireec_flags[i] = ""
        if flag not in [None, "", " "]:
            flags.append(flag)

    if target_triple in ["gfx940", "gfx941", "gfx942", "gfx90a"]:
        if "unet" in safe_name:
            flags.extend(gfx94X_flags["unet"])
        if any(x in safe_name for x in ["clip", "prompt_encoder"]):
            flags.extend(gfx94X_flags["clip"])
        if "vae" in safe_name:
            flags.extend(gfx94X_flags["vae"])
        flags.extend(gfx94X_flags["all"])

    if attn_spec is not None:
        flags.extend(["--iree-codegen-transform-dialect-library=" + attn_spec])

    print("Compiling to", device, "with flags:", flags)

    if mlir_source == "file":
        flatbuffer_blob = ireec.compile_file(
            module_str,
            target_backends=[device],
            input_type="torch",
            extra_args=flags,
        )
    elif mlir_source == "str":
        if save_mlir:
            with open(f"{safe_name}.mlir", "w+") as f:
                f.write(module_str)
            print("Saved to", safe_name + ".mlir")
        flatbuffer_blob = ireec.compile_str(
            module_str,
            target_backends=[device],
            input_type="torch",
            extra_args=flags,
        )
    else:
        raise ValueError("mlir_source must be either 'file' or 'str'")
    with open(f"{safe_name}.vmfb", "wb+") as f:
        f.write(flatbuffer_blob)
    print("Saved to", safe_name + ".vmfb")
    if return_path == True:
        return safe_name + ".vmfb"


def create_safe_name(hf_model_name, model_name_str):
    safe_name = hf_model_name.split("/")[-1].strip() + model_name_str
    safe_name = re.sub("-", "_", safe_name)
    safe_name = re.sub("\.", "_", safe_name)
    return safe_name


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
            if external_weight_file and not os.path.isfile(external_weight_file):
                safetensors.torch.save_file(mod_params, external_weight_file)
                print("Saved params to", external_weight_file)


def largest_error(array1, array2):
    absolute_diff = np.abs(array1 - array2)
    max_error = np.max(absolute_diff)
    print("Max error:", max_error)
    return max_error


def get_schedulers(model_id):
    # TODO: Robust scheduler setup on pipeline creation -- if we don't
    # set batch_size here, the SHARK schedulers will
    # compile with batch size = 1 regardless of whether the model
    # outputs latents of a larger batch size, e.g. SDXL.
    # However, obviously, searching for whether the base model ID
    # contains "xl" is not very robust.

    batch_size = 2 if "xl" in model_id.lower() else 1

    schedulers = dict()
    schedulers["PNDM"] = PNDMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    return schedulers
