# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import turbine_models.custom_models.stateless_llama as llama
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
from transformers.modeling_utils import load_sharded_checkpoint
import copy
import torch

os.environ["TORCH_LOGS"] = "dynamic"
from shark_turbine.aot import *
from turbine_models.custom_models import llm_runner

from turbine_models.gen_external_params.gen_external_params import (
    gen_external_params,
)
from turbine_models.turbine_tank import turbine_tank


DEFAULT_PROMPT = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
"""


def main():
    hf_model_name = "Trelis/Llama-2-7b-chat-hf-function-calling-v2"
    hf_auth_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float16,
        token=hf_auth_token,
    )
    device = "cuda"
    model.to(device)
    torch_str = llm_runner.run_torch_llm(
        hf_model_name,
        None,
        copy.copy(DEFAULT_PROMPT),
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    torch.cuda.empty_cache()
    rotated_torch_str = llm_runner.run_torch_llm(
        hf_model_name,
        None,
        copy.copy(DEFAULT_PROMPT),
        streaming_llm=True,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    print("REF:", torch_str)
    print("Rotated:", rotated_torch_str)


if __name__ == "__main__":
    main()
