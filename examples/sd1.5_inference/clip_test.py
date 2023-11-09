# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import torch._dynamo as dynamo
from torch._export import dynamic_dim
from torch._export.constraints import constrain_as_size, constrain_as_value
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer"
)
text_encoder_model = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="text_encoder"
)

def main():
    # Example input values
    prompt = ["a photograph of an astronaut riding a horse"]
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    print('INPUT IDS:', text_input.input_ids, text_input.input_ids.shape, text_input.input_ids.dtype) # 1x77

    outputs = text_encoder_model(text_input.input_ids)

    opt = torch.compile(text_encoder_model, backend="turbine_cpu")
    opt(text_input.input_ids)


class ModelTests(unittest.TestCase):
    def testCLIP(self):
        main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()