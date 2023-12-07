# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import turbine_models.custom_models.stateless_llama as llama
import unittest
import os
import pytest

from typing import Literal

def test_export(quantization: Literal["int4", None], precision: Literal["f16", "f32"]):
   
    llama.export_transformer_model(
        hf_model_name="llSourcell/medllama2_7b",
        hf_auth_token=None,
        compile_to="vmfb",
        external_weights="safetensors",
        # external_weight_file="medllama2_7b_f16_int4.safetensors", Do not export weights because this doesn't get quantized
        quantization=quantization,
        precision=precision,
        device="llvm-cpu",
        target_triple="host",
        max_alloc = "4294967296"
    )

    from turbine_models.gen_external_params.gen_external_params import gen_external_params
    gen_external_params(
        hf_model_name="llSourcell/medllama2_7b",
        quantization=quantization,
        weight_path="medllama2_7b_f16_int4.safetensors",
        hf_auth_token=None,
        precision=precision
    )

    from types import SimpleNamespace
    args = SimpleNamespace()
    args.hf_model_name = "llSourcell/medllama2_7b"
    args.hf_auth_token = None
    args.vmfb_path = "medllama2_7b.vmfb"
    args.external_weight_file = "medllama2_7b_f16_int4.safetensors"
    args.run_vmfb = True
    args.device="llvm-cpu"
    args.precision = precision
    args.quantization = quantization
    args.iree_target_triple="host"
    llama.run_vmfb_comparison(args)
    
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
