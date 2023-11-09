# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import shark_turbine.aot as aot
import torch
import torch._dynamo as dynamo
from torch._export import dynamic_dim
from torch._export.constraints import constrain_as_size, constrain_as_value
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

# Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer"
)
text_encoder_model = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="text_encoder"
)
example_x = torch.empty(1, 77, dtype=torch.int64)
exported = aot.export(text_encoder_model, example_x)
exported.print_readable()
compiled_binary = exported.compile(save_to=None)


def infer():
    import numpy as np
    import iree.runtime as rt

    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    prompt = ["a photograph of an astronaut riding a horse"]
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    inp = text_input.input_ids
    outputs = vmm.main(inp)
    for output in outputs:
        print(output.to_host(), output.to_host().shape)


class ModelTests(unittest.TestCase):
    def testCLIP(self):
        infer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()