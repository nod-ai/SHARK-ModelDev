from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import safetensors
from iree.compiler.ir import Context
import torch
import shark_turbine.aot as aot
from shark_turbine.aot import *

class HFTransformerBuilder:
    """
    A model builder that uses Hugging Face's transformers library to build a PyTorch model.

    Args:
        example_input (torch.Tensor): An example input tensor to the model.
        hf_id (str): The Hugging Face model ID.
        auto_model (AutoModel): The AutoModel class to use for loading the model.
        auto_tokenizer (AutoTokenizer): The AutoTokenizer class to use for loading the tokenizer.
        auto_config (AutoConfig): The AutoConfig class to use for loading the model configuration.
    """

    def __init__(
        self,
        example_input: torch.Tensor,
        hf_id: str,
        auto_model: AutoModel = AutoModelForCausalLM,
        auto_tokenizer: AutoTokenizer = AutoTokenizer,
        auto_config: AutoConfig = None,
        hf_auth_token="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",
    ) -> None:
        self.example_input = example_input
        self.hf_id = hf_id
        self.auto_model = auto_model
        self.auto_tokenizer = auto_tokenizer
        self.auto_config = auto_config
        self.hf_auth_token = hf_auth_token
        self.model = None
        self.tokenizer = None
        self.build_model()

    def build_model(self) -> None:
        """
        Builds a PyTorch model using Hugging Face's transformers library.
        """
        # TODO: check cloud storage for existing ir
        self.model = self.auto_model.from_pretrained(
            self.hf_id, token=self.hf_auth_token, torch_dtype=torch.float, trust_remote_code=True
        )
        #if self.auto_tokenizer is not None:
        #    self.tokenizer = self.auto_tokenizer.from_pretrained(
        #        self.hf_id, token=self.hf_auth_token, use_fast=False
        #    )
        #else:
        self.tokenizer = None

    def get_compiled_module(self, save_to: str = None) -> aot.CompiledModule:
        """
        Compiles the PyTorch model into a compiled module using SHARK-Turbine's AOT compiler.

        Args:
            save_to (str): one of: input (Torch IR) or import (linalg).

        Returns:
            aot.CompiledModule: The compiled module binary.
        """
        module = aot.export(self.model, self.example_input)
        compiled_binary = module.compile(save_to=save_to)
        return compiled_binary


if __name__ == "__main__":
    import sys
    hf_id = sys.argv[-1]
    safe_name = hf_id.replace("/", "_").replace("-", "_")
    inp = torch.zeros(1, 1, dtype=torch.int64)
    model = HFTransformerBuilder(inp, hf_id)
    mapper=dict()
    mod_params = dict(model.model.named_parameters())
    for name in mod_params:
        mapper["params." + name] = name
#        safetensors.torch.save_file(mod_params, safe_name+".safetensors")
    class GlobalModule(CompiledModule):
        params = export_parameters(model.model, external=True, external_scope="",)
        compute = jittable(model.model.forward)

        def run(self, x=aot.AbstractTensor(1, None, dtype=torch.int64)):
            return self.compute(x, constraints=[
                    x.dynamic_dim(1),]
                )

        def run_not(self, x=abstractify(inp)):
            return self.compute(x)

    print("module defined")
    inst = GlobalModule(context=Context())
    print("module inst")
    module = CompiledModule.get_mlir_module(inst)
#    compiled = module.compile()
    print("got mlir module")
    with open(safe_name+".mlir", "w+") as f:
        f.write(str(module))

    print("done")