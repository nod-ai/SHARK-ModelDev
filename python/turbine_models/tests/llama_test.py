# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import turbine_models.custom_models.stateless_llama as llama
import unittest
import os


class LLamaTest(unittest.TestCase):
    def testExportTransformerModel(self):
        llama.export_transformer_model(
            "meta-llama/Llama-2-7b-chat-hf",
            # TODO: replace with github secret
            "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",
            "torch",
            "safetensors",
            "llama_f32.safetensors",
            None,
            "f32",
        )
        os.remove("llama_f32.safetensors")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
