import argparse
import os
from pathlib import Path


def path_expand(s):
    return Path(s).expanduser().resolve()


def is_valid_file(arg):
    if not os.path.exists(arg):
        return None
    else:
        return arg


# Note: this is where command-line options for the scripts in this directory
# are defined along with their defaults. Thus, they should not be referenced
# within modelling or inference code, only at the entry point to the script.

# We should consider separating out the options that are "model configs" from
# the options that control the compiler, runtime, and script behavior,
# when applicable, as the former would best be kept in a separate
# config or imported from huggingface.

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

##############################################################################
# SDXL Huggingface Options
##############################################################################

p.add_argument(
    "--hf_auth_token",
    type=str,
    help="The Hugging Face auth token, if required",
    default=None,
)
p.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="stabilityai/stable-diffusion-2-1",
)
p.add_argument(
    "--model_arch",
    type=str,
    help="SD pipeline/model architecture. Choices are [sd, sdxl, sd3].",
    default=None,
)
p.add_argument(
    "--scheduler_id",
    type=str,
    help="Scheduler ID",
    default="EulerDiscrete",
)

##############################################################################
# SDXL Inference Options
#    These options are used to control runtime parameters for SDXL inference.
##############################################################################

p.add_argument(
    "--prompt",
    type=str,
    default=" a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal",
    help="Prompt input to stable diffusion.",
)

p.add_argument(
    "--negative_prompt",
    type=str,
    default="Watermark, blurry, oversaturated, low resolution, pollution",
    help="Negative prompt input to stable diffusion.",
)

p.add_argument(
    "--num_inference_steps", type=int, default=30, help="Number of UNet inference steps"
)

p.add_argument(
    "--batch_count",
    type=int,
    default=1,
    help="Number of batches to run for a single prompt",
)

p.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help="Scale by which to adjust prompt guidance to the unconditional noise prediction output of UNet after each iteration.",
)

p.add_argument(
    "--seed", type=float, default=0, help="Seed for random number/latents generation."
)

p.add_argument(
    "--external_weight_path",
    type=str,
    default="",
    help="Path to external weights file, for jobs with one weights filepath. When importing, this is used to specify where to save the model weights, and at runtime, this is used to specify where to load the model weights from.",
)

p.add_argument(
    "--external_weights_dir",
    type=str,
    default="./weights",
    help="Directory containing external weights for a job that requires more than one weights file. When importing, this is used to specify where to save the model weights, and at runtime, this is used to specify where to load the model weights from. Files will then be saved according to the parameters that make them unique, i.e. <hf_model_name>_<precision>_<submodel>_<submodel-specific>.<external_weights>",
)

p.add_argument(
    "--vmfb_path", type=str, default="", help="path to vmfb containing compiled module"
)

p.add_argument(
    "--pipeline_vmfb_path",
    type=str,
    default="",
    help="path to vmfb containing compiled meta-module",
)

p.add_argument(
    "--external_weight_file",
    type=str,
    default=None,
    help="Path to external weights, used in benchmark scripts.",
)

p.add_argument(
    "--pipeline_dir",
    type=str,
    default="./vmfbs",
    help="Directory to save pipeline artifacts",
)

p.add_argument(
    "--compiled_pipeline",
    default=False,
    action="store_true",
    help="Do one-shot inference from tokens to image in a shrink-wrapped pipeline binary.",
)

p.add_argument(
    "--cpu_scheduling",
    default=False,
    action="store_true",
    help="Run scheduling on native pytorch CPU backend.",
)

p.add_argument(
    "--benchmark",
    type=str,
    default=None,
    help="A comma-separated list of submodel IDs for which to report benchmarks for, or 'all' for all components.",
)

p.add_argument(
    "--save_outputs",
    type=str,
    default=None,
    help="A comma-separated list of submodel IDs for which to save output .npys for, or 'all' for all components.",
)
##############################################################################
# SDXL Modelling Options
#    These options are used to control model defining parameters for SDXL.
#    These are MLIR - changing variables! If you change them, you will need
#    to import/download and recompile the model.
##############################################################################

p.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
p.add_argument(
    "--height", type=int, default=512, help="Height of Stable Diffusion output image."
)
p.add_argument(
    "--width", type=int, default=512, help="Width of Stable Diffusion output image"
)
p.add_argument(
    "--precision",
    type=str,
    default="fp16",
    help="Precision of Stable Diffusion weights and graph.",
)

p.add_argument(
    "--clip_precision",
    type=str,
    default=None,
    help="Precision of CLIP weights and graph.",
)
p.add_argument(
    "--unet_precision",
    type=str,
    default=None,
    help="Precision of UNet weights and graph.",
)
p.add_argument(
    "--mmdit_precision",
    type=str,
    default=None,
    help="Precision of mmdit weights and graph.",
)
p.add_argument(
    "--vae_precision",
    type=str,
    default=None,
    help="Precision of  vae weights and graph.",
)

p.add_argument(
    "--clip_spec",
    type=str,
    default=None,
    help="transform dialect spec for the given submodel.",
)
p.add_argument(
    "--unet_spec",
    type=str,
    default=None,
    help="transform dialect spec for the given submodel.",
)
p.add_argument(
    "--mmdit_spec",
    type=str,
    default=None,
    help="transform dialect spec for the given submodel.",
)
p.add_argument(
    "--vae_spec",
    type=str,
    default=None,
    help="transform dialect spec for the given submodel.",
)


p.add_argument(
    "--max_length", type=int, default=64, help="Sequence Length of Stable Diffusion"
)

p.add_argument(
    "--decomp_attn",
    default=False,
    action="store_true",
    help="Decompose attention at fx graph level",
)

p.add_argument(
    "--clip_decomp_attn",
    action="store_true",
    help="Decompose attention for text_encoder only at fx graph level",
)

p.add_argument(
    "--vae_decomp_attn",
    action="store_true",
    help="Decompose attention for VAE decode only at fx graph level",
)

p.add_argument(
    "--unet_decomp_attn",
    action="store_true",
    help="Decompose attention for unet only at fx graph level",
)

p.add_argument(
    "--mmdit_decomp_attn",
    action="store_true",
    help="Decompose attention for unet only at fx graph level",
)


p.add_argument(
    "--use_i8_punet",
    action="store_true",
    help="Use i8 quantized Partitioned UNet for inference",
)

##############################################################################
# SDXL script general options.
##############################################################################

p.add_argument("--compile_to", type=str, default="mlir", help="torch, linalg, vmfb")

p.add_argument("--verbose", "-v", action="store_true")
p.add_argument(
    "--external_weights",
    type=str,
    default=None,
    choices=["safetensors", "irpa", "gguf", None],
    help="Externalizes model weights from the torch dialect IR and its successors",
)

# See --external_weight_path and --external_weight_dir to specify where to save the model weights.

p.add_argument(
    "--compare_vs_torch",
    action="store_true",
    help="Runs both turbine vmfb and a torch model to compare results",
)
p.add_argument(
    "--exit_on_vmfb",
    default=True,
    action="store_false",
    help="Exit program on vmfb compilation completion. Most scripts will also save .mlir if this is disabled.",
)
p.add_argument(
    "--input_mlir",
    type=str,
    default=None,
    help="Path to input mlir file to compile. Comma-separate paths to provide more than one input to pipelines.",
)
p.add_argument(
    "--download_mlir",
    default=False,
    action="store_true",
    help="Download missing mlir files from Azure storage.",
)
p.add_argument(
    "--container_name",
    type=str,
    default=None,
    help="Azure storage container name to download mlir files from.",
)


##############################################################################
# IREE Compiler Options
##############################################################################

p.add_argument(
    "--device",
    type=str,
    default="local-task",
    help="local-task, local-sync, vulkan://0, rocm://0, cuda://0, etc.",
)

p.add_argument(
    "--clip_device",
    type=str,
    default=None,
    help="local-task, local-sync, vulkan://0, rocm://0, cuda://0, etc.",
)
p.add_argument(
    "--unet_device",
    type=str,
    default=None,
    help="local-task, local-sync, vulkan://0, rocm://0, cuda://0, etc.",
)
p.add_argument(
    "--mmdit_device",
    type=str,
    default=None,
    help="local-task, local-sync, vulkan://0, rocm://0, cuda://0, etc.",
)
p.add_argument(
    "--vae_device",
    type=str,
    default=None,
    help="local-task, local-sync, vulkan://0, rocm://0, cuda://0, etc.",
)
p.add_argument(
    "--scheduler_device",
    type=str,
    default=None,
    help="local-task, local-sync, vulkan://0, rocm://0, cuda://0, etc.",
)

# TODO: Bring in detection for target triple
p.add_argument(
    "--iree_target_triple",
    type=str,
    default="x86_64-linux-gnu",
    help="Specify vulkan target triple or rocm/cuda target chip.",
)

p.add_argument(
    "--clip_target",
    type=str,
    default=None,
    help="Specify vulkan target triple or rocm/cuda target chip.",
)
p.add_argument(
    "--unet_target",
    type=str,
    default=None,
    help="Specify vulkan target triple or rocm/cuda target chip.",
)
p.add_argument(
    "--mmdit_target",
    type=str,
    default=None,
    help="Specify vulkan target triple or rocm/cuda target chip.",
)
p.add_argument(
    "--vae_target",
    type=str,
    default=None,
    help="Specify vulkan target triple or rocm/cuda target chip.",
)
p.add_argument(
    "--scheduler_target",
    type=str,
    default=None,
    help="Specify vulkan target triple or rocm/cuda target chip.",
)

p.add_argument("--ireec_flags", type=str, default="", help="extra iree-compile options")

p.add_argument(
    "--attn_flags",
    type=str,
    default="",
    help="extra iree-compile options for models with iree_linalg_ext.attention ops.",
)

p.add_argument(
    "--clip_flags",
    type=str,
    default="",
    help="extra iree-compile options to send for compiling CLIP/prompt_encoder. Only use this for testing bleeding edge flags! Any default options should be added to sd_inference/utils.py",
)

p.add_argument(
    "--vae_flags",
    type=str,
    default="",
    help="extra iree-compile options to send for compiling VAE. Only use this for testing bleeding edge flags! Any default options should be added to sd_inference/utils.py",
)

p.add_argument(
    "--unet_flags",
    type=str,
    default="",
    help="extra iree-compile options to send for compiling unet. Only use this for testing bleeding edge flags! Any default options should be added to sd_inference/utils.py",
)

p.add_argument(
    "--mmdit_flags",
    type=str,
    default="",
    help="extra iree-compile options to send for compiling mmdit. Only use this for testing bleeding edge flags! Any default options should be added to sd_inference/utils.py",
)

p.add_argument(
    "--attn_spec",
    type=str,
    default=None,
    help="extra iree-compile options for models with iree_linalg_ext.attention ops. Set this to 'default' if you are using mfma-capable hardware with ROCM.",
)


args, unknown = p.parse_known_args()
