from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import shark_turbine.aot as aot
from turbine_models.turbine_tank import turbine_tank
import os
import re


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
        auto_model: AutoModel = AutoModel,
        auto_tokenizer: AutoTokenizer = None,
        auto_config: AutoConfig = None,
        hf_auth_token=None,
        upload_ir=False,
        model=None,
        model_type: str = None,
        run_e2e: bool = None,
    ) -> None:
        self.example_input = example_input
        self.hf_id = hf_id
        self.auto_model = auto_model
        self.auto_tokenizer = auto_tokenizer
        self.auto_config = auto_config
        self.hf_auth_token = hf_auth_token
        self.model = model
        self.tokenizer = None
        self.upload_ir = upload_ir
        self.model_type = model_type
        self.run_e2e = run_e2e
        if self.model == None:
            self.build_model()

    def build_model(self) -> None:
        """
        Builds a PyTorch model using Hugging Face's transformers library.
        """
        # TODO: check cloud storage for existing ir
        self.model = self.auto_model.from_pretrained(
            self.hf_id, token=self.hf_auth_token, config=self.auto_config
        )
        if self.auto_tokenizer is not None:
            self.tokenizer = self.auto_tokenizer.from_pretrained(
                self.hf_id, token=self.hf_auth_token
            )
        else:
            self.tokenizer = None

    def get_compiled_module(self, save_to: str = None) -> aot.CompiledModule:
        """
        Compiles the PyTorch model into a compiled module using SHARK-Turbine's AOT compiler.

        Args:
            save_to (str): one of: input (Torch IR) or import (linalg).

        Returns:
            aot.CompiledModule: The compiled module binary.
        """
        if self.model_type == "hf_seq2seq":
            module = aot.export(self.model, *self.example_input)
        else:
            module = aot.export(self.model, self.example_input)
        module_str = str(module.mlir_module)
        safe_name = self.hf_id.split("/")[-1].strip()
        safe_name = re.sub("-", "_", safe_name)
        if self.upload_ir:
            with open(f"{safe_name}.mlir", "w+") as f:
                f.write(module_str)
            model_name_upload = self.hf_id.replace("/", "_")
            turbine_tank.uploadToBlobStorage(
                str(os.path.abspath(f"{safe_name}.mlir")),
                f"{model_name_upload}/{model_name_upload}.mlir",
            )
            os.remove(f"{safe_name}.mlir")
        if self.run_e2e is not None and self.run_e2e is False:
            return
        compiled_binary = module.compile(save_to=save_to)
        return compiled_binary
