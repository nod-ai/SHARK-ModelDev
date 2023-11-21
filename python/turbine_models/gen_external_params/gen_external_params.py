import re
from turbine_models.model_builder import HFTransformerBuilder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name ID",
    default="meta-llama/Llama-2-7b-chat-hf",
)
parser.add_argument("--quantization", type=str, default="int4")
parser.add_argument("--weight_path", type=str, default="")
parser.add_argument(
    "--hf_auth_token", type=str, help="The HF auth token required for some models"
)
parser.add_argument(
    "--precision", type=str, default="f16", help="Data type of model [f16, f32]"
)


def quantize(model, quantization, dtype):
    accumulates = dtype
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

        int_weights = {}
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

    all_weights.update(int_weights)
    return all_weights


if __name__ == "__main__":
    args = parser.parse_args()
    model_builder = HFTransformerBuilder(
        example_input=None,
        hf_id=args.hf_model_name,
        auto_model=AutoModelForCausalLM,
        hf_auth_token=args.hf_auth_token,
    )
    model_builder.build_model()
    if args.precision == "f16":
        model = model_builder.model.half()
        dtype = torch.float16
    elif args.precision == "f32":
        model = model_builder.model
        dtype = torch.float32
    else:
        sys.exit("invalid precision, f16 or f32 supported")
    quant_weights = quantize(model, args.quantization, dtype)
    # TODO: Add more than just safetensor support
    import safetensors

    if args.weight_path == "":
        save_path = args.hf_model_name.split("/")[-1].strip()
        save_path = re.sub("-", "_", save_path)
        save_path = (
            save_path + "_" + args.precision + "_" + args.quantization + ".safetensors"
        )
    else:
        save_path = args.weight_path
    safetensors.torch.save_file(quant_weights, save_path)
    print("Saved safetensor output to ", save_path)
