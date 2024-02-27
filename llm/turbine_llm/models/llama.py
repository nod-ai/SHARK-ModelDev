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
    LlamaAttentionBlock,
    RotaryEmbeddingLayer,
)

from ..layers.core import (
    LinearLayer,
    RMSNormLayer,
    ThetaLayer,
    TokenEmbedding,
)


class DirectCacheLlamaModelV1(ThetaLayer):
    """Simple LlamaModel with a direct lookup KV cache."""

    def __init__(self, theta: Theta, hp: LlamaHParams):
        super().__init__(theta)
        self.hp = hp
        self.add_module("token_embedding", TokenEmbedding(theta("token_embd")))
        self.add_module(
            "attention_embedding",
            RotaryEmbeddingLayer(
                rope_dimension_count=hp.rope_dimension_count,
                max_seqlen=hp.context_length,
            ),
        )
        self.add_module(
            "output_norm",
            RMSNormLayer(
                theta("output_norm"), epsilon=self.hp.attention_layer_norm_rms_epsilon
            ),
        )
        self.add_module("output_lm_head", LinearLayer(theta("output")))

        self.attn_blocks = nn.ModuleList(
            [
                LlamaAttentionBlock(
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

    def forward(
        self,
        tokens: torch.Tensor,
        start_index: int,
        *,
        return_logits: bool = False,
        local_kv_cache: list[torch.Tensor]
    ):
        bs, sl = tokens.shape
        h = self.token_embedding(tokens)
        dtype = h.dtype

        # Compute attention mask.
        attention_mask = None
        if sl > 1:
            # Use the smallest value like HF as opposed to -inf like original.
            # A little bit easier for some systems.
            attention_mask = torch.full(
                (1, 1, sl, sl), torch.finfo(dtype).min, dtype=dtype
            )
            attention_mask = torch.triu(
                attention_mask, diagonal=start_index + 1
            ).type_as(h)

        # Iterate over attention blocks.
        block_count = len(self.attn_blocks)
        for block_idx, block in enumerate(self.attn_blocks):
            block_cache_k = local_kv_cache[block_idx]
            block_cache_v = local_kv_cache[block_count + block_idx]
            block_output = block(
                h,
                cache_k=block_cache_k,
                cache_v=block_cache_v,
                start_index=start_index,
                attention_mask=attention_mask,
            )
            h = h + block_output

        h = self.output_norm(h)
        logits = self.output_lm_head(h)

        if return_logits:
            return h
        else:
            last_step = logits[:, -1, :]
            token = torch.argmax(last_step, keepdim=True, dim=1)
            return token.to(tokens.dtype)
