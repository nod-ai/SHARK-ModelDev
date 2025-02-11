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
# general options
##############################################################################

p.add_argument(
    "--hf_auth_token",
    type=str,
    help="The Hugging Face auth token, if required",
    default=None,
)
p.add_argument(
    "--model_id",
    type=str,
    help="model ID as it appears in the torchbench models text file lists, or 'all' for batch export",
    default="all",
)
p.add_argument(
    "--model_lists",
    type=Path,
    nargs="*",
    help="path to a JSON list of models to benchmark. One or more paths.",
    default=["torchbench_models.json", "timm_models.json", "torchvision_models.json"],
)
p.add_argument(
    "--external_weights_dir",
    type=str,
    default="",
    help="Path to external weights file, for jobs with one weights filepath. When importing, this is used to specify where to save the model weights, and at runtime, this is used to specify where to load the model weights from.",
)
p.add_argument(
    "--vmfbs_dir", type=str, default="", help="path to vmfb containing compiled module"
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
p.add_argument("--compile_to", type=str, default="mlir", help="torch, linalg, vmfb")
p.add_argument(
    "--external_weights",
    type=str,
    default="irpa",
    choices=["safetensors", "irpa", "gguf", None],
    help="Externalizes model weights from the torch dialect IR and its successors",
)
p.add_argument(
    "--run_benchmark",
    type=bool,
    default=True,
)
p.add_argument(
    "--num_iters",
    type=int,
    default=10,
)
p.add_argument(
    "--output_csv",
    type=str,
    default="./benchmark_results.csv",
)

##############################################################################
# Modeling and Export Options
#    These options are used to control model defining parameters.
#    These are MLIR - changing variables! If you change them, you will need
#    to import/download and recompile the model.
##############################################################################

p.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
p.add_argument(
    "--precision",
    type=str,
    default="fp16",
    help="Precision of Stable Diffusion weights and graph.",
)
p.add_argument(
    "--decomp_attn",
    default=False,
    action="store_true",
    help="Decompose attention at fx graph level",
)

# See --external_weight_path and --external_weight_dir to specify where to save the model weights.

p.add_argument(
    "--compare_vs_torch",
    action="store_true",
    help="Runs both turbine vmfb and a torch model to compare results",
)
p.add_argument(
    "--input_mlir",
    type=str,
    default=None,
    help="Path to input mlir file to compile. Comma-separate paths to provide more than one input to pipelines.",
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
    "--target",
    type=str,
    default="gfx942",
    help="Usually a rocm chip arch or llvmcpu target triple, e.g. gfx942 or x86_64-linux-gnu.",
)
p.add_argument("--ireec_flags", type=str, default="", help="extra iree-compile options")
p.add_argument(
    "--attn_spec",
    type=str,
    default=None,
    help="extra iree-compile options for models with sdpa ops.",
)


args, unknown = p.parse_known_args()
