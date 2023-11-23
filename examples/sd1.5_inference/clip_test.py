# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import numpy as np
import shark_turbine.aot as aot
import torch
import torch._dynamo as dynamo
from torch._export import dynamic_dim
from torch._export.constraints import constrain_as_size, constrain_as_value
from transformers import CLIPTextModel, CLIPTokenizer

pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"


# Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer"
)
text_encoder_model = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="text_encoder"
)
class CompiledUnet(aot.CompiledModule):
    params = aot.export_parameters(text_encoder_model)

    def main(self, inp=aot.AbstractTensor(1, 77, dtype=torch.int64)):
        return aot.jittable(text_encoder_model.forward)(
            inp
        )


exported = aot.export(CompiledUnet)
compiled_binary = exported.compile(save_to=None)
prompt = ["a photograph of an astronaut riding a horse"]
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
inp = text_input.input_ids


def infer():
    import iree.runtime as rt

    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    outputs = vmm.main(inp)
    output = outputs[0]
    print('TURBINE:', output.to_host(), output.to_host().shape, output.to_host().dtype)
    return output


def infer_torch():
    torch_output = text_encoder_model.forward(inp)[0]
    np_torch_output = torch_output.detach().cpu().numpy()
    print('TORCH:', np_torch_output, np_torch_output.shape, np_torch_output.dtype)
    return np_torch_output


def largest_error(array1, array2):
    absolute_diff = np.abs(array1 - array2)
    max_error = np.max(absolute_diff)
    return max_error


class ModelTests(unittest.TestCase):
    def testCLIP(self):
        torch_output = infer_torch()
        turbine_output = infer()
        err = largest_error(torch_output, turbine_output)
        print('LARGEST ERROR:', err)
        assert(err < 9e-5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()