# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from turbine_models.gen_external_params.gen_external_params import quantize
from turbine_models.model_builder import HFTransformerBuilder
from transformers import AutoTokenizer, AutoModelForCausalLM
import unittest
import os
import torch
import pytest


class ExternalParamsTest(unittest.TestCase):
    def testQuantizeF32(self):
        model_builder = HFTransformerBuilder(
            example_input=None,
            hf_id="facebook/opt-350m",
            auto_model=AutoModelForCausalLM,
        )
        model_builder.build_model()
        quant_weights = quantize(model_builder.model, "", torch.float32)
        for weight in quant_weights:
            self.assertNotIn("weight_zp", weight)
            self.assertNotIn("weight_scale", weight)
            assert quant_weights[weight].dtype in [torch.float32]

    def testQuantizeF32I8(self):
        model_builder = HFTransformerBuilder(
            example_input=None,
            hf_id="facebook/opt-350m",
            auto_model=AutoModelForCausalLM,
        )
        model_builder.build_model()
        quant_weights = quantize(model_builder.model, "int8", torch.float32)
        named_params = dict(model_builder.model.named_parameters())
        for weight in quant_weights:
            if "weight_scale" not in weight and "weight_zp" not in weight:
                if "layers" in weight and "weight" in weight and "norm" not in weight:
                    assert quant_weights[weight].dtype in [torch.uint8]
                    assert named_params[weight].size(dim=1) == quant_weights[
                        weight
                    ].size(dim=1)
                else:
                    assert quant_weights[weight].dtype in [torch.float32]
            else:
                assert quant_weights[weight].dtype in [torch.float32]

    def testQuantizeF32I4(self):
        model_builder = HFTransformerBuilder(
            example_input=None,
            hf_id="facebook/opt-350m",
            auto_model=AutoModelForCausalLM,
        )
        model_builder.build_model()
        quant_weights = quantize(model_builder.model, "int4", torch.float32)
        named_params = dict(model_builder.model.named_parameters())
        for weight in quant_weights:
            if "weight_scale" not in weight and "weight_zp" not in weight:
                if "layers" in weight and "weight" in weight and "norm" not in weight:
                    assert quant_weights[weight].dtype in [torch.uint8]
                    assert named_params[weight].size(dim=1) == 2 * quant_weights[
                        weight
                    ].size(dim=1)
                else:
                    assert quant_weights[weight].dtype in [torch.float32]
            else:
                assert quant_weights[weight].dtype in [torch.float32]

    def testQuantizeF16(self):
        model_builder = HFTransformerBuilder(
            example_input=None,
            hf_id="facebook/opt-350m",
            auto_model=AutoModelForCausalLM,
        )
        model_builder.build_model()
        quant_weights = quantize(model_builder.model.half(), "", torch.float16)
        for weight in quant_weights:
            self.assertNotIn("weight_zp", weight)
            self.assertNotIn("weight_scale", weight)
            assert quant_weights[weight].dtype in [torch.float16]

    @pytest.mark.xfail(reason="brevitas issue with f16 int8 quanttensor")
    def testQuantizeF16I8(self):
        model_builder = HFTransformerBuilder(
            example_input=None,
            hf_id="facebook/opt-350m",
            auto_model=AutoModelForCausalLM,
        )
        model_builder.build_model()
        quant_weights = quantize(model_builder.model.half(), "int8", torch.float16)
        named_params = dict(model_builder.model.named_parameters())
        for weight in quant_weights:
            if "weight_scale" not in weight and "weight_zp" not in weight:
                if "layers" in weight and "weight" in weight and "norm" not in weight:
                    assert quant_weights[weight].dtype in [torch.uint8]
                    assert named_params[weight].size(dim=1) == quant_weights[
                        weight
                    ].size(dim=1)
                else:
                    assert quant_weights[weight].dtype in [torch.float16]
            else:
                assert quant_weights[weight].dtype in [torch.float16]

    def testQuantizeF16I4(self):
        model_builder = HFTransformerBuilder(
            example_input=None,
            hf_id="facebook/opt-350m",
            auto_model=AutoModelForCausalLM,
        )
        model_builder.build_model()
        quant_weights = quantize(model_builder.model.half(), "int4", torch.float16)
        named_params = dict(model_builder.model.named_parameters())
        for weight in quant_weights:
            if "weight_scale" not in weight and "weight_zp" not in weight:
                if "layers" in weight and "weight" in weight and "norm" not in weight:
                    assert quant_weights[weight].dtype in [torch.uint8]
                    assert named_params[weight].size(dim=1) == 2 * quant_weights[
                        weight
                    ].size(dim=1)
                else:
                    assert quant_weights[weight].dtype in [torch.float16]
            else:
                assert quant_weights[weight].dtype in [torch.float16]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
