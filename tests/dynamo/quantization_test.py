import logging
import unittest
import pytest

from shark_turbine.dynamo.importer import FxImporter
from shark_turbine.dynamo.passes import turbine_cpu_pass_pipeline
import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd
from torch.fx import (
    GraphModule,
)

from torch._decomp import get_decompositions
import numpy as np

from torch.func import functionalize
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class optInferenceModel():#torch.nn.Module):
    def __init__(
        self,
        device,
        base_model_name = "facebook/opt-125m"
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=self.tokenizer)#BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config = self.quantization_config).to(device=self.device)
        #examples = [self.tokenizer("Sample input for quantization")]
        #self.base_model.quantize(examples)

    def set_device(self, device):
        self.device = device
        self.base_model.to(self.device)

    def save(self, path):
        self.base_model.save(self.base_model.state_dict(), path)#save_quantized(path)
    
    def forward(self, prompt):
        return self.tokenizer.decode(self.base_model.generate(**self.tokenizer(prompt, return_tensors="pt").to(self.device))[0])
    #def get_inputs(self, prompt):
    #    return self.tokenizer(prompt, return_tensors="pt").to(device=self.device)

def infer_iteration(model, prompt):
    return model.forward(prompt)

class test_autogptq_int4_opt(unittest.TestCase):
#    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
#    torch._dynamo.config.verbose=True
#    opt_model_id = "facebook/opt-125m"
#    tokenizer = AutoTokenizer.from_pretrained(opt_model_id)
#    quantizati on_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
    #model = AutoModelForCausalLM.from_pretrained(opt_model_id, quantization_config=quantization_config).to(device="cuda:0")#, quantization_config)#, device_map="auto")#, quantization_config=quantization_config)#.to(device="cpu")
    #model.save("opt_quant_model")
    #exit(0)
    model = optInferenceModel("cuda:0") #AutoGPTQForCausalLM.from_quantized("opt_quant_model", device="cpu")#optInferenceModel("cuda:0")   
    prompt = "What is a llama?"
#    inputs = tokenizer(prompt, return_tensors="pt").to(device="cuda:0")
    #inputs = model.get_inputs(prompt)
    print(prompt)
    golden_quant_out = model.forward(prompt)
    print("golden output: ", golden_quant_out)
    
    #model.base_model.to(device="cpu")#set_device("cpu")
    model.set_device("cpu")
    #turbine_model = optInferenceModel("cpu")
    #opt_mod = torch.compile(forward, backend="turbine.cpu")
    opt_mod = torch.compile(infer_iteration, backend="turbine_cpu") 
    #inputs = opt_mod.get_inputs(prompt)
    output = opt_mod(model, prompt)
    print("Turbine output: ", output)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()  
