from transformers import AutoModelForCausalLM
import safetensors
from iree.compiler.ir import Context
import torch
import shark_turbine.aot as aot
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="bert-large-uncased",
)
parser.add_argument(
    "--hf_auth_token",
    type=str,
    help="The Hugging Face auth token, required",
)
parser.add_argument("--compile_to", type=str, default="linalg", help="linalg, vmfb")
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="saves ir/vmfb without global weights for size and readability, options [gguf, safetensors]",
)
parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, vulkan, rocm")
# TODO: Bring in detection for target triple
parser.add_argument(
    "--iree_target_triple",
    type=str,
    default="host",
    help="Specify vulkan target triple or rocm/cuda target device.",
)
parser.add_argument("--vulkan_max_allocation", type=str, default="4294967296")


def export_bert_model(
    hf_model_name,
    hf_auth_token=None,
    external_weights=None,
    compile_to="linalg",
    device=None,
    target_triple=None,
    max_alloc=None,
):
    safe_name = args.hf_model_name.split("/")[-1].strip().replace("-", "_")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        token=hf_auth_token,
        torch_dtype=torch.float,
        trust_remote_code=True,
    )

    mapper = {}
    if external_weights is not None:
        if external_weights == "safetensors":
            mod_params = dict(model.named_parameters())
            for name in mod_params:
                mapper["params." + name] = name
            safetensors.torch.save_file(mod_params, safe_name + ".safetensors")

        elif external_weights == "gguf":
            tensor_mapper = remap_gguf.TensorNameMap(remap_gguf.MODEL_ARCH.LLAMA, HEADS)
            mapper = tensor_mapper.mapping

    class BertModule(CompiledModule):
        if external_weights:
            params = export_parameters(
                model, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(model)
        compute = jittable(model.forward)

        def run_forward(
            self,
            x=AbstractTensor(1, 1, dtype=torch.int64),
            mask=AbstractTensor(1, 1, dtype=torch.int64),
        ):
            return self.compute(x, attention_mask=mask)

    inst = BertModule(context=Context())
    module_str = str(CompiledModule.get_mlir_module(inst))

    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(module_str)
    print("Saved to", safe_name + ".mlir")

    if compile_to == "vmfb":
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)


if __name__ == "__main__":
    args = parser.parse_args()
    export_bert_model(
        args.hf_model_name,
        args.hf_auth_token,
        args.external_weights,
        args.compile_to,
        args.device,
        args.iree_target_triple,
        args.vulkan_max_allocation,
    )
