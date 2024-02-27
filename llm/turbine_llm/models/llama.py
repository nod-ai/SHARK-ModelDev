# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn as nn

from ..config.llm_configs import LlamaHParams
from ..data import Theta
from ..layers.attention import (
    LlamaAttentionLayer,
    RotaryEmbeddingLayer,
)

from ..layers.core import (
    ThetaLayer,
)


class LlamaModelV1(ThetaLayer):
    """Simple LlamaModel with a direct lookup KV cache."""

    def __init__(self, theta: Theta, hp: LlamaHParams):
        super().__init__(theta)
        self.hp = hp
        self.add_module(
            "attention_embedding",
            RotaryEmbeddingLayer(
                rope_dimension_count=hp.rope_dimension_count,
                max_seqlen=hp.context_length,
            ),
        )
        self.attn_blocks = nn.ModuleList(
            [
                LlamaAttentionLayer(
                    theta("blk", n),
                    embedding=self.attention_embedding,
                    head_count=hp.attention_head_count,
                    head_dim=hp.rope_dimension_count,
                    head_count_kv=hp.attention_head_count_kv,
                    rms_epsilon=hp.attention_layer_norm_rms_epsilon,
                )
                for n in range(hp.block_count)
            ]
        )
