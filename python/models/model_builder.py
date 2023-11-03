from transformers import AutoModel, AutoTokenizer, AutoConfig
from abc import ABC, abstractmethod
import torch
import shark_turbine.aot as aot


class ModelBuilder(ABC):
    """
    Abstract base class for building compiled turbine modules.
    """
    def __init__(self, example_input:torch.Tensor = None) -> None:
        super().__init__()
        self.example_input = example_input
        self.compiled_binary = None

    def build_model(self) -> torch.nn.Module:
        """
        Abstract method for building a PyTorch model.
        """
        self.example_input = None
        self.model = None
        pass

    def get_compiled_module(self, save_to: str = None) -> aot.CompiledModule:
        """
        Compiles the PyTorch model into a compiled module using SHARK-Turbine's AOT compiler.

        Args:
            save_to (str): one of: input (Torch IR) or import (linalg).

        Returns:
            aot.CompiledModule: The compiled module binary.
        """
        if self.compiled_binary is None:
            #TODO: check cloud storage for existing ir
            module = aot.export(self.build_model(), self.example_input)
            self.compiled_binary = module.compile(save_to=save_to)
        return self.compiled_binary
    
            
    def infer(self) -> None:
        """
        Abstract method for running inference on the compiled module.
        """
        pass



    

class HFBuilder(ModelBuilder):
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
        super().__init__(example_input)
        self.hf_id = hf_id
        self.auto_model = auto_model
        self.auto_tokenizer = auto_tokenizer
        self.auto_config = auto_config
        self.build_model(example_input)

    def build_model(self) -> None:
        """
        Builds a PyTorch model using Hugging Face's transformers library.
        """
        self.model = self.auto_model.from_pretrained(self.hf_id, config=self.auto_config)
        if self.auto_tokenizer is not None:
            self.tokenizer = self.auto_tokenizer.from_pretrained(self.hf_id)
        else:
            self.tokenizer = None
         

# this is currrently irrelevant because you can just do aot.export(model, example_input)        
class TorchModuleBuilder(ModelBuilder):
    """
    A model builder that uses a pre-built PyTorch model.

    Args:
        example_input (torch.Tensor): An example input tensor to the model.
        model (torch.nn.Module): The pre-built PyTorch model.
    """
    def __init__(self, example_input: torch.Tensor, model: torch.nn.Module) -> None:
        self.model = model
        self.example_input = example_input
        self.build_model(example_input)
        
    def build_model(self) -> None:
        pass




#TODO: pipeline builder, other model types, quantization, global handling once supported by aot.export api