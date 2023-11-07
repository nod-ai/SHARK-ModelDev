from transformers import AutoModel, AutoTokenizer, AutoConfig
from abc import ABC, abstractmethod
import torch
import shark_turbine.aot as aot

class HFBuilder():
    """
    A model builder that uses Hugging Face's transformers library to build a PyTorch model.

    Args:
        example_input (torch.Tensor): An example input tensor to the model.
        hf_id (str): The Hugging Face model ID.
        auto_model (AutoModel): The AutoModel class to use for loading the model.
        auto_tokenizer (AutoTokenizer): The AutoTokenizer class to use for loading the tokenizer.
        auto_config (AutoConfig): The AutoConfig class to use for loading the model configuration.
    """
    def __init__(self, example_input: torch.Tensor, hf_id: str, auto_model: AutoModel = AutoModel, auto_tokenizer: AutoTokenizer = None, auto_config: AutoConfig = None ) -> None:
        self.example_input = example_input
        self.hf_id = hf_id
        self.auto_model = auto_model
        self.auto_tokenizer = auto_tokenizer
        self.auto_config = auto_config
        self.model = None
        self.tokenizer = None
        self.build_model()

    def build_model(self) -> None:
        """
        Builds a PyTorch model using Hugging Face's transformers library.
        """
        #TODO: check cloud storage for existing ir
        self.model = self.auto_model.from_pretrained(self.hf_id, config=self.auto_config)
        if self.auto_tokenizer is not None:
            self.tokenizer = self.auto_tokenizer.from_pretrained(self.hf_id)
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
        module = aot.export(self.model, self.example_input)
        compiled_binary = module.compile(save_to=save_to)
        return compiled_binary