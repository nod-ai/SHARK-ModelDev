from urllib.request import urlopen
import iree.compiler as ireec
import numpy as np
import os
import safetensors
import safetensors.numpy as safe_numpy
import re
import glob
from diffusers import (
    PNDMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    # DPMSolverSDEScheduler,
)

# If flags are verified to work on a specific model and improve performance without regressing numerics, add them to this dictionary. If you are working with bleeding edge flags, please add them manually with the --ireec_flags argument.
MI_flags = {
    "all": [
        "--iree-global-opt-propagate-transposes=true",
        "--iree-opt-const-eval=false",
        "--iree-llvmgpu-enable-prefetch=true",
        "--iree-execution-model=async-external",
    ],
    "pad_attention": [
        "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-global-opt-raise-special-ops, iree-preprocessing-pad-to-intrinsics, util.func(iree-linalg-ext-pad-attention{pad-to-multiple-of=0,128,0,32,0}))",
    ],
    "punet": [
        "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-dispatch-creation-canonicalize), iree-preprocessing-transpose-convolution-pipeline,  iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))"
    ],
    "vae_preprocess": [
        "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-dispatch-creation-canonicalize), iree-preprocessing-transpose-convolution-pipeline,  iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))"
    ],
    "preprocess_default": [
        "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics{pad-target-type=conv})",
    ],
    "unet": [
        "--iree-dispatch-creation-enable-aggressive-fusion",
        "--iree-dispatch-creation-enable-fuse-horizontal-contractions=false",
        "--iree-opt-aggressively-propagate-transposes=true",
        "--iree-codegen-llvmgpu-use-vector-distribution=true",
        "--iree-opt-data-tiling=false",
        "--iree-vm-target-truncate-unsupported-floats",
        "--iree-opt-outer-dim-concat=true",
        "--iree-codegen-gpu-native-math-precision=true",
        "--iree-hal-indirect-command-buffers=true",
        "--iree-stream-resource-memory-model=discrete",
        "--iree-hal-memoization=true",
        "--iree-opt-strip-assertions",
    ],
    "clip": [
        "--iree-dispatch-creation-enable-aggressive-fusion",
        "--iree-dispatch-creation-enable-fuse-horizontal-contractions=false",
        "--iree-opt-aggressively-propagate-transposes=true",
        "--iree-opt-outer-dim-concat=true",
        "--iree-hip-waves-per-eu=2",
        "--iree-codegen-llvmgpu-use-vector-distribution=true",
    ],
    "vae": [
        "--iree-dispatch-creation-enable-aggressive-fusion",
        "--iree-dispatch-creation-enable-fuse-horizontal-contractions=false",
        "--iree-opt-aggressively-propagate-transposes=true",
        "--iree-codegen-llvmgpu-use-vector-distribution=true",
        "--iree-opt-data-tiling=false",
        "--iree-codegen-gpu-native-math-precision=true",
        "--iree-vm-target-truncate-unsupported-floats",
    ],
    "winograd": [""],
}
GFX11_flags = {
    "all": [
        "--iree-global-opt-propagate-transposes=true",
        "--iree-opt-outer-dim-concat=true",
        "--iree-vm-target-truncate-unsupported-floats",
        "--iree-llvmgpu-enable-prefetch=true",
        "--iree-opt-data-tiling=false",
        "--iree-opt-const-eval=false",
        "--iree-opt-aggressively-propagate-transposes=true",
        "--iree-dispatch-creation-enable-aggressive-fusion",
        "--iree-dispatch-creation-enable-fuse-horizontal-contractions=false",
        "--iree-codegen-gpu-native-math-precision=true",
        "--iree-codegen-llvmgpu-use-vector-distribution=true",
    ],
    "pad_attention": [
        "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-global-opt-raise-special-ops, iree-preprocessing-pad-to-intrinsics, util.func(iree-linalg-ext-pad-attention{pad-to-multiple-of=0,64,0,32,0}))",
    ],
    "punet": [
        "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-dispatch-creation-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))"
        "--iree-dispatch-creation-enable-fuse-horizontal-contractions=true",
        "--iree-codegen-llvmgpu-enable-transform-dialect-jit=false",
    ],
    "preprocess_default": [
        "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-global-opt-raise-special-ops, iree-preprocessing-pad-to-intrinsics)",
        "--iree-codegen-llvmgpu-enable-transform-dialect-jit=false",
    ],
    "unet": [""],
    "clip": [""],
    "vae": [""],
    "winograd": [""],
}
znver4_flags = {
    "all": [
        "--iree-llvmcpu-target-cpu=znver4",
        "--iree-opt-const-eval=false",
        "--iree-llvmcpu-enable-ukernels=mmt4d,pack,unpack",
        "--iree-dispatch-creation-collapse-reduction-dims",
        "--iree-opt-const-expr-max-size-increase-threshold=1000000000000000",
        "--iree-dispatch-creation-enable-fuse-padding-into-linalg-consumer-ops",
    ],
    "bf16": [
        "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-demote-contraction-inputs-to-bf16))",
    ],
    "winograd": [
        "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-linalg-ext-convert-conv2d-to-winograd{replace-all-convs=true},iree-global-opt-demote-contraction-inputs-to-bf16))"
    ],
}

_IREE_DRIVER_MAP = {
    "cpu": "local-task",
    "cpu-task": "local-task",
    "cpu-sync": "local-sync",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "metal",
    "rocm": "hip",
    "rocm-legacy": "rocm",
    "hip": "hip",
    "intel-gpu": "level_zero",
}

_IREE_BACKEND_MAP = {
    "cpu": "llvm-cpu",
    "local-task": "llvm-cpu",
    "local-sync": "llvm-cpu",
    "rocm": "rocm",
    "rocm-legacy": "rocm",
    "hip": "rocm",
    "cuda": "cuda",
    "vulkan": "vulkan-spirv",
    "metal": "metal",
}


def iree_device_map(device):
    uri_parts = device.split("://", 2)
    iree_driver = (
        _IREE_DRIVER_MAP[uri_parts[0]]
        if uri_parts[0] in _IREE_DRIVER_MAP
        else uri_parts[0]
    )
    if len(uri_parts) == 1:
        return iree_driver
    else:
        return f"{iree_driver}://{uri_parts[1]}"


def iree_backend_map(device):
    uri_parts = device.split("://", 2)
    iree_device = (
        _IREE_BACKEND_MAP[uri_parts[0]]
        if uri_parts[0] in _IREE_BACKEND_MAP
        else uri_parts[0]
    )
    return iree_device


def replace_with_tk_kernels(tk_kernels_dir, flow_dialect_ir, batch_size):
    kernels = glob.glob(tk_kernels_dir + "/bs" + str(batch_size) + "/*")

    # Replace all calls to old kernel with new kernel
    print("Inserting kernels and updating calls to kernels...")
    kernel_name = {}
    for kernel in kernels:
        kernel_name[kernel] = kernel.split("/")[-1].split(".")[0]
    kernel_map = {}
    prefix_map = {}

    base = flow_dialect_ir.split("\n")
    new_base = []
    for line in base:
        for kernel in kernels:
            suffix = kernel.split(".")[0].split("_")[-1]
            if "bias" in suffix:
                suffix = kernel.split(".")[0].split("_")[-2]
            B, M, N, K = suffix.split("x")
            old_kernel = f"matmul_like_{B}x{M}x{N}x{K}"
            if not old_kernel in line:
                continue
            if old_kernel in line and "func.func" in line:
                num_args = line.count("arg")
                with open(kernel, "r") as f:
                    data = f.readlines()
                idx_with_kernel_args = [
                    idx for idx, s in enumerate(data) if "func.func" in s
                ][0]
                kernel_args = data[idx_with_kernel_args].count("arg")
                if num_args != kernel_args:
                    continue
                kernel_map[kernel] = line.strip().split(" ")[1][1:-7]
                prefix_map[kernel] = kernel_map[kernel].split(old_kernel)[0][:-1]
            if (
                old_kernel in line
                and "flow.dispatch" in line
                and not "func.func" in line
            ):
                line = line.replace(kernel_map[kernel], kernel_name[kernel])
                line = line.replace(prefix_map[kernel], kernel_name[kernel])
        new_base.append(line)
    # Insert kernels in appropriate locations
    final_ir = []
    for line in new_base:
        for kernel in kernels:
            if (
                prefix_map[kernel] + " {" in line
                and "flow.executable" in line
                and "private" in line
            ):
                with open(kernel, "r") as f:
                    data = f.readlines()
                translation_info = data[0].split("#translation = ")[1].strip()
                extract = "".join(data[2:-2])
                extract = extract.replace("#translation", translation_info)
                final_ir += extract
        final_ir.append(line)

    print("tk kernels added")
    return final_ir


def compile_to_vmfb(
    module_str,
    device,
    target_triple,
    ireec_flags=[""],
    safe_name="model",
    return_path=False,
    const_expr_hoisting=True,
    mlir_source="str",
    max_alloc="4294967296",
    save_mlir=True,
    attn_spec=None,
    winograd=False,
    flagset_keywords=[],
    debug=False,
    add_tk_kernels=False,
    tk_kernels_dir=None,
    batch_size=1,
):
    if batch_size != 1 and batch_size != 8:
        add_tk_kernels = False
    flags = []
    if mlir_source == "file" and not isinstance(module_str, str):
        module_str = str(module_str)
    if target_triple in ["", None]:
        if device == "cpu":
            target_triple = "x86_64-linux-gnu"
        else:
            raise ValueError(
                "target_triple must be set. Usually this can be fixed by setting --iree_target_triple in the CLI."
            )
    if device in ["cpu", "llvm-cpu"]:
        if target_triple == "znver4":
            flags.extend(znver4_flags["all"])
            if winograd:
                flags.extend(znver4_flags["winograd"])
        else:
            flags.extend(
                [
                    "--iree-llvmcpu-target-triple=" + target_triple,
                    "--iree-llvmcpu-target-cpu-features=host",
                    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false",
                    "--iree-llvmcpu-distribution-size=32",
                    "--iree-opt-const-eval=false",
                    "--iree-llvmcpu-enable-ukernels=all",
                    "--iree-global-opt-enable-quantized-matmul-reassociation",
                ]
            )
        device = "llvm-cpu"
    elif device in ["vulkan", "vulkan-spirv"]:
        flags.extend(
            [
                "--iree-hal-target-backends=vulkan-spirv",
                "--iree-vulkan-target-triple=" + target_triple,
                "--iree-stream-resource-max-allocation-size=" + max_alloc,
                "--iree-stream-resource-index-bits=64",
                "--iree-vm-target-index-bits=64",
                "--iree-dispatch-creation-inline-constants-max-byte-length=1",
            ]
        )
        device = "vulkan-spirv"
    elif device in ["rocm", "hip"]:
        flags.extend(
            [
                "--iree-hal-target-backends=rocm",
                "--iree-hip-target=" + target_triple,
                "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
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
    if isinstance(ireec_flags, str):
        if ireec_flags != "":
            ireec_flags = ireec_flags.split(",")
    elif ireec_flags == None:
        ireec_flags = []

    if debug:
        flags.extend(
            ["--iree-hal-dump-executable-files-to=" + safe_name + "_dispatches"]
        )

    if target_triple in ["gfx940", "gfx941", "gfx942", "gfx90a"]:
        if "unet" in safe_name:
            flags.extend(MI_flags["unet"])
        elif any(x in safe_name for x in ["clip", "prompt_encoder"]):
            flags.extend(MI_flags["clip"])
        elif "vae" in safe_name:
            flags.extend(MI_flags["vae"])
        flags.extend(MI_flags["all"])
        if "masked_attention" in flagset_keywords:
            flags.extend(MI_flags["pad_attention"])
        elif "punet" in flagset_keywords:
            flags.extend(MI_flags["punet"])
        elif "vae" in safe_name:
            flags.extend(MI_flags["vae_preprocess"])
        else:
            flags.extend(MI_flags["preprocess_default"])

    if "gfx11" in target_triple:
        flags.extend(GFX11_flags["all"])
        if "masked_attention" in flagset_keywords:
            flags.extend(GFX11_flags["pad_attention"])
        elif "punet" in flagset_keywords:
            flags.extend(GFX11_flags["punet"])
        else:
            flags.extend(GFX11_flags["preprocess_default"])

    # Currently, we need a transform dialect script to be applied to the compilation through IREE in certain cases.
    # This 'attn_spec' handles a linalg_ext.attention op lowering to mfma instructions for capable targets.
    # This is a temporary solution, and should be removed or largely disabled once the functionality of
    # the TD spec is implemented in C++.

    if attn_spec in ["default", "mfma", "punet"]:
        if any(x in safe_name for x in ["clip", "prompt_encoder"]) == False:
            use_punet = True if attn_spec in ["punet", "i8"] else False
            attn_spec = get_mfma_spec_path(
                target_triple,
                os.path.dirname(safe_name),
                use_punet=use_punet,
            )
            flags.extend(["--iree-codegen-transform-dialect-library=" + attn_spec])

    elif attn_spec in ["wmma"] or ("gfx11" in target_triple and not attn_spec):
        attn_spec = get_wmma_spec_path(target_triple, os.path.dirname(safe_name))
        if attn_spec:
            flags.extend(["--iree-codegen-transform-dialect-library=" + attn_spec])
    elif attn_spec and attn_spec != "None":
        if any(x in safe_name for x in ["clip", "prompt_encoder"]) == False:
            flags.extend(["--iree-codegen-transform-dialect-library=" + attn_spec])

    for i, flag in enumerate(ireec_flags):
        k = flag.strip().split("=")[0]
        for idx, default in enumerate(flags):
            if default == None:
                flags.pop(idx)
                continue
            elif k == default.split("=")[0]:
                flags[idx] = flag if flag.split("=")[-1] not in ["None", ""] else None
                flag = None
                if flags[idx] == None:
                    flags.pop(idx)
                continue
        if flag not in [None, "", " "] and flag.split("=")[-1] not in ["None", ""]:
            flags.append(flag)

    for idx, flag in enumerate(flags):
        if flag is None:
            flags.pop(idx)
    input_ir_type = "torch"
    if add_tk_kernels:
        print("Adding tk kernels")
        flags.extend(["--compile-to=flow"])
        if mlir_source == "file":
            flatbuffer_blob = ireec.compile_file(
                module_str,
                target_backends=[device],
                input_type=input_ir_type,
                extra_args=flags,
            )
        elif mlir_source == "str":
            flatbuffer_blob = ireec.compile_str(
                module_str,
                target_backends=[device],
                input_type=input_ir_type,
                extra_args=flags,
            )

        flow_ir = flatbuffer_blob.decode("utf-8")

        flow_ir_tk = replace_with_tk_kernels(tk_kernels_dir, flow_ir, batch_size)
        module_str = "\n".join(flow_ir_tk)
        flags.pop()
        flags.extend(["--compile-from=flow"])
        mlir_source = "str"
        input_ir_type = "auto"

    print("Compiling to", device, "with flags:", flags)

    # Forces a standard for naming files:
    # If safe_name has target triple in it, get rid of target triple in mlir name
    #
    if target_triple not in safe_name:
        safe_vmfb_name = safe_name + "_" + target_triple
        safe_mlir_name = safe_name
    else:
        safe_vmfb_name = safe_name
        safe_mlir_name = "".join(safe_name.split(target_triple))

    if mlir_source == "file":
        flatbuffer_blob = ireec.compile_file(
            module_str,
            target_backends=[device],
            input_type=input_ir_type,
            extra_args=flags,
        )
    elif mlir_source == "str":
        if save_mlir:
            with open(f"{safe_mlir_name}.mlir", "w+") as f:
                f.write(module_str)
            print("Saved to", safe_mlir_name + ".mlir")
        flatbuffer_blob = ireec.compile_str(
            module_str,
            target_backends=[device],
            input_type=input_ir_type,
            extra_args=flags,
        )
    else:
        raise ValueError("mlir_source must be either 'file' or 'str'")
    with open(f"{safe_vmfb_name}.vmfb", "wb+") as f:
        f.write(flatbuffer_blob)
    print(f"Saved to {safe_vmfb_name}.vmfb")
    if return_path == True:
        return safe_vmfb_name + ".vmfb"


def create_safe_name(hf_model_name, model_name_str=""):
    if not model_name_str:
        model_name_str = ""
    if model_name_str != "" and (not model_name_str.startswith("_")):
        model_name_str = "_" + model_name_str

    safe_name = hf_model_name.split("/")[-1].strip() + model_name_str
    safe_name = re.sub("-", "_", safe_name)
    safe_name = re.sub("\.", "_", safe_name)
    return safe_name


def get_mfma_spec_path(target_chip, save_dir, masked_attention=False, use_punet=False):
    if use_punet:
        suffix = "_punet"
        url = "https://raw.githubusercontent.com/iree-org/iree/refs/heads/main/build_tools/pkgci/external_test_suite/attention_and_matmul_spec_punet_mi300.mlir"
    elif not masked_attention:
        suffix = ""
        url = "https://raw.githubusercontent.com/iree-org/iree/refs/heads/main/build_tools/pkgci/external_test_suite/attention_and_matmul_spec.mlir"
    else:
        suffix = "_pad"
        url = "https://sharkpublic.blob.core.windows.net/sharkpublic/specs/latest/attention_and_matmul_spec_gfx942.mlir"
    attn_spec = urlopen(url).read().decode("utf-8")
    spec_path = os.path.join(save_dir, f"attention_and_matmul_spec_mfma{suffix}.mlir")
    with open(spec_path, "w") as f:
        f.write(attn_spec)
    return spec_path


def get_wmma_spec_path(target_chip, save_dir, masked_attention=False):
    if not masked_attention:
        url = "https://sharkpublic.blob.core.windows.net/sharkpublic/specs/no_pad/attention_and_matmul_spec_wmma.mlir"
    elif target_chip == "gfx1100":
        url = "https://sharkpublic.blob.core.windows.net/sharkpublic/specs/latest/attention_and_matmul_spec_gfx1100.mlir"
    elif target_chip in ["gfx1103", "gfx1150"]:
        url = "https://sharkpublic.blob.core.windows.net/sharkpublic/specs/latest/attention_and_matmul_spec_gfx1150.mlir"
    else:
        return None
    attn_spec = urlopen(url).read().decode("utf-8")
    suffix = "masked" if masked_attention else ""
    spec_path = os.path.join(save_dir, f"attention_and_matmul_spec_wmma{suffix}.mlir")
    with open(spec_path, "w") as f:
        f.write(attn_spec)
    return spec_path


def save_external_weights(
    mapper,
    model,
    external_weights=None,
    external_weight_file=None,
    force_format=False,
    vae_harness=False,
):
    if external_weights is not None:
        if external_weights in ["safetensors", "irpa"]:
            mod_params = dict(model.named_parameters())
            mod_buffers = dict(model.named_buffers())
            mod_params.update(mod_buffers)
            vae_params = {}
            for name in mod_params:
                if vae_harness:
                    vae_params[name.replace("vae.", "")] = mod_params[name]
                mapper["params." + name] = name
            if vae_harness:
                mod_params = vae_params
            if external_weight_file and not os.path.isfile(external_weight_file):
                if not force_format:
                    safetensors.torch.save_file(mod_params, external_weight_file)
                else:
                    for x in mod_params.keys():
                        mod_params[x] = mod_params[x].numpy()
                    safe_numpy.save_file(mod_params, external_weight_file)
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
    schedulers["EulerDiscrete"] = EulerDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["EulerAncestralDiscrete"] = (
        EulerAncestralDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
        )
    )
    # schedulers["DPMSolverSDE"] = DPMSolverSDEScheduler.from_pretrained(
    #     model_id,
    #     subfolder="scheduler",
    # )
    return schedulers