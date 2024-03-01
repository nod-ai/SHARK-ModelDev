# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.llm_configs import LlamaHParams
from ..data import Theta
from ..layers import (
    LinearLayer,
    RMSNormLayer,
    ThetaLayer,
    TokenEmbedding,
    RotaryEmbeddingLayer,
)

__all__ = [
    "DirectCacheLlamaModelV1",
]

################################################################################
# Models
################################################################################


class DirectCacheLlamaModelV1(ThetaLayer):
    """Simple LlamaModel with a direct lookup KV cache for batch-1 inference."""

    def __init__(self, theta: Theta, hp: LlamaHParams):
        super().__init__(theta)
        self.hp = hp
        self.add_module(
            "token_embedding",
            TokenEmbedding(theta("token_embd"), dtype=hp.activation_dtype),
        )
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

    def create_cache(self, bs: int) -> list[torch.Tensor]:
        return [
            torch.empty(
                (
                    bs,
                    self.hp.context_length,
                    self.hp.attention_head_count,
                    self.hp.rope_dimension_count,
                ),
                dtype=self.hp.activation_dtype,
            )
            for _ in range(self.hp.block_count * 2)
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        start_index: int,
        *,
        return_logits: bool = False,
        local_kv_cache: list[torch.Tensor],
    ):
        bs, sl = tokens.shape
        h = self.token_embedding(tokens)
        dtype = h.dtype
        self.trace_tensor("llama.token_embedding", h)

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
            if block_idx == 0:
                self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
            h = block(
                h,
                cache_k=block_cache_k,
                cache_v=block_cache_v,
                start_index=start_index,
                attention_mask=attention_mask,
            )
            self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)

        if return_logits:
            return h
        else:
            last_step = logits[:, -1, :]
            token = torch.argmax(last_step, keepdim=True, dim=1)
            return token.to(tokens.dtype)


################################################################################
# Layers
################################################################################


class LlamaAttentionBlock(ThetaLayer):
    """Implements a self attention layer in the style of Llama."""

    def __init__(
        self,
        theta: Theta,
        *,
        head_count: int,
        head_dim: int,
        head_count_kv: int,
        embedding: RotaryEmbeddingLayer,
        rms_epsilon: float,
    ):
        super().__init__(theta)
        self.add_module(
            "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=rms_epsilon)
        )
        self.add_module("attn_q", LinearLayer(theta("attn_q")))
        self.add_module("attn_k", LinearLayer(theta("attn_k")))
        self.add_module("attn_v", LinearLayer(theta("attn_v")))
        self.add_module("attn_output", LinearLayer(theta("attn_output")))
        self.add_module(
            "ffn_norm", RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
        )
        self.add_module("ffn_gate", LinearLayer(theta("ffn_gate")))
        self.add_module("ffn_up", LinearLayer(theta("ffn_up")))
        self.add_module("ffn_down", LinearLayer(theta("ffn_down")))

        self.embedding = embedding
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv

    def forward(
        self,
        h: torch.Tensor,
        *,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        start_index: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        x = self.attn_norm(h)

        bs, q_len, feature_dim = x.shape
        kv_seq_len = start_index + q_len
        assert feature_dim == self.head_count * self.head_dim

        xq = self.attn_q(x)
        xk = self.attn_k(x)
        xv = self.attn_v(x)

        xq = xq.view(bs, q_len, self.head_count, self.head_dim)
        xk = xk.view(bs, q_len, self.head_count_kv, self.head_dim)
        xv = xv.view(bs, q_len, self.head_count_kv, self.head_dim)

        xq, xk = self.embedding(xq=xq, xk=xk, start_index=start_index)

        # TODO: Some model variants do some form of kv repetition to expand the
        # count of kv heads to the count of attention heads used by the q.
        assert self.head_count == self.head_count_kv, "NYI: KV expansion"

        # Update our positions in the cache.
        cache_k[:bs, start_index:kv_seq_len] = xk
        cache_v[:bs, start_index:kv_seq_len] = xv

        # Derive keys/values from the entirety of the available sequence.
        keys = cache_k[:bs, :kv_seq_len]
        values = cache_v[:bs, :kv_seq_len]

        # Tranpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Flash attention.
        attn_weights = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply attention mask.
        if attention_mask is not None:
            expected_mask_shape = (bs, 1, q_len, kv_seq_len)
            assert (
                attention_mask.shape == expected_mask_shape
            ), f"Attention mask should be of size {expected_mask_shape}, but is {attention_mask.shape}"
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(xq)
        attn_output = torch.matmul(attn_weights, values)  # (bs, heads, slen, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bs, q_len, -1)

        # Project.
        attn_output = self.attn_output(attn_output)

        # Remainder of the block.
        h = h + attn_output

        # Feed forward network.
        ffn_input = self.ffn_norm(h)
        ffn_gate = F.silu(self.ffn_gate(ffn_input))
        ffn_up = self.ffn_up(ffn_input)
        ffn_down = self.ffn_down(ffn_gate * ffn_up)
        return h + ffn_down
