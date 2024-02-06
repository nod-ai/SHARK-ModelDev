import re
from typing import Literal
from turbine_models.model_builder import HFTransformerBuilder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import argparse
import sys

parser = argparse.ArgumentParser(description="Quantize and save Hugging Face models.")

parser.add_argument(
    "--hf_model_name",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="The Hugging Face model name ID.",
)
parser.add_argument(
    "--quantization",
    type=str,
    default="int4",
    choices=["unquantized", "int4", "int8"],
    help="Type of quantization to apply.",
)
parser.add_argument(
    "--weight_path",
    type=str,
    default="",
    help="Path to save the quantized model weights.",
)
parser.add_argument(
    "--hf_auth_token",
    type=str,
    default=None,
    help="The Hugging Face auth token required for some models.",
)
parser.add_argument(
    "--precision",
    type=str,
    default="f16",
    choices=["f16", "f32"],
    help="Data type of model.",
)


def quantize(model, quantization, dtype):
    accumulates = dtype
    int_weights = {}
    if quantization in ["int4", "int8"]:
        from brevitas_examples.common.generative.quantize import quantize_model
        from brevitas_examples.llm.llm_quant.run_utils import get_model_impl

        print("Applying weight quantization...")
        weight_bit_width = 4 if quantization == "int4" else 8
        quantize_model(
            get_model_impl(model).layers,
            dtype=accumulates,
            weight_bit_width=weight_bit_width,
            weight_param_method="stats",
            weight_scale_precision="float_scale",
            weight_quant_type="asym",
            weight_quant_granularity="per_group",
            weight_group_size=128,  # TODO: make adjustable
            quantize_weight_zero_point=False,
        )
        from brevitas_examples.llm.llm_quant.export import LinearWeightBlockQuantHandler
        from brevitas.nn.quant_linear import QuantLinear

        class DummyLinearWeightBlockQuantHandler(LinearWeightBlockQuantHandler):
            def forward(self, x):
                raise NotImplementedError

        for prefix, layer in model.named_modules():
            if isinstance(layer, QuantLinear):
                print(f"Exporting layer {prefix}")
                exporter = DummyLinearWeightBlockQuantHandler()
                exporter.prepare_for_export(layer)
                print(
                    f"  weight = ({exporter.int_weight.shape}, {exporter.int_weight.dtype}), "
                    f"scale=({exporter.scale.shape}, {exporter.scale.dtype}), "
                    f"zero=({exporter.zero_point.shape}, {exporter.zero_point.dtype})"
                )
                int_weights[f"{prefix}.weight"] = exporter.int_weight
                int_weights[f"{prefix}.weight_scale"] = exporter.scale
                int_weights[f"{prefix}.weight_zp"] = exporter.zero_point

    all_weights = dict(model.named_parameters())
    for k in list(all_weights.keys()):
        if "wrapped_scaling_impl" in k or "wrapped_zero_point_impl" in k:
            del all_weights[k]

    if len(int_weights) != 0:
        all_weights.update(int_weights)
    return all_weights


def gen_external_params(
    hf_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    quantization: Literal["unquantized", "int4", "int8"] = "int4",
    weight_path: str = "",
    hf_auth_token: str = None,
    precision: str = "f16",
):
    """
    Main function to run the model quantization and saving process.

    :param hf_model_name: The Hugging Face model name ID.
    :param quantization: Type of quantization to apply ('int4' or 'int8').
    :param weight_path: Path to save the quantized model weights.
    :param hf_auth_token: The Hugging Face auth token required for some models.
    :param precision: Data type of model ('f16' or 'f32').
    """
    SUPPORTED_QUANTIZATIONS = ["unquantized", "int4", "int8"]
    if quantization not in SUPPORTED_QUANTIZATIONS:
        if (
            quantization is None
            or quantization.lower() == "none"
            or quantization.lower() == "unquantized"
        ):
            quantization = "unquantized"
        else:
            raise ValueError(f"Invalid quantization, {quantization} not supported.")

    model_builder = HFTransformerBuilder(
        example_input=None,
        hf_id=hf_model_name,
        auto_model=AutoModelForCausalLM,
        hf_auth_token=hf_auth_token,
    )

    if precision == "f16":
        model = model_builder.model.half()
        dtype = torch.float16
    elif precision == "f32":
        model = model_builder.model
        dtype = torch.float32
    else:
        sys.exit("Invalid precision, f16 or f32 supported")

    quant_weights = quantize(model, quantization, dtype)

    if weight_path == "":
        save_path = hf_model_name.split("/")[-1].strip()
        save_path = re.sub("-", "_", save_path)
        save_path = save_path + "_" + precision + "_" + quantization + ".safetensors"
    else:
        save_path = weight_path

    import safetensors

    safetensors.torch.save_file(quant_weights, save_path)
    print("Saved safetensor output to ", save_path)


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        gen_external_params(
            hf_model_name=args.hf_model_name,
            quantization=args.quantization,
            weight_path=args.weight_path,
            hf_auth_token=args.hf_auth_token,
            precision=args.precision,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
