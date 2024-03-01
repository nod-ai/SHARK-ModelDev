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
    PagedKVCache,
)

from .base import BaseCausalLMModel

__all__ = [
    "PagedLlamaModelV1",
]


################################################################################
# Models
################################################################################


class PagedLlamaModelV1(BaseCausalLMModel):
    """LlamaModel with a paged KV cache and supporting variable sequence
    length batched inference.

    As both the caching and batching setup is complicated, this model variant
    is modular, intending to be instantiated and used in an overall assembly
    vs trying to providing one-stop methods that do everything.

    The inference procedure is typically:

    1. Initialize the PagedKVCache state tensors.
    2. Generate an input mask given a vector of sequence lengths.
    3. Generate an attention mask from the input mask.
    4. Allocate a block mapping table.
    5. Invoke prefill() with a batch of sequences.
    6. Extract tokens from batched logits.
    7. Iteratively invoke decode() for as long as there are sequences needing
       to be serviced.

    Various samplers and schedulers can be interleaved throughout.
    """

    def __init__(self, theta: Theta, hp: LlamaHParams):
        super().__init__(theta, context_length=hp.context_length)
        self.hp = hp
        # TODO: It doesn't seem like this is the right way to be getting the
        # innermost attention dim.
        attn_head_dim = hp.rope_dimension_count
        self.cache = PagedKVCache(
            transformer_block_count=hp.block_count,
            attn_head_count=hp.attention_head_count_kv,
            attn_head_dim=attn_head_dim,
            cache_partition_count=2,
            block_seq_stride=16,
        )
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
                PagedLlamaAttentionBlock(
                    theta("blk", n),
                    block_index=n,
                    cache=self.cache,
                    embedding=self.attention_embedding,
                    head_count=hp.attention_head_count,
                    head_dim=attn_head_dim,
                    head_count_kv=hp.attention_head_count_kv,
                    rms_epsilon=hp.attention_layer_norm_rms_epsilon,
                )
                for n in range(hp.block_count)
            ]
        )

    def prefill(
        self,
        # [bs, batch_seq_len]
        tokens: torch.Tensor,
        *,
        # [1, 1, batch_seq_len, batch_seq_len]
        attention_mask: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        cache_state: list[torch.Tensor],
    ):
        h = self.token_embedding(tokens)
        self.trace_tensor("llama.token_embedding", h)

        # Iterate over attention blocks.
        for block_idx, block in enumerate(self.attn_blocks):
            self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
            h = block(
                h,
                start_index=0,
                attention_mask=attention_mask,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
            )
            self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)
        return logits


################################################################################
# Layers
################################################################################


class PagedLlamaAttentionBlock(ThetaLayer):
    """Implements a self attention layer in the style of Llama using a
    paged cache."""

    def __init__(
        self,
        theta: Theta,
        *,
        block_index: int,
        cache: PagedKVCache,
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

        self.block_index = block_index
        self.cache = cache
        self.embedding = embedding
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv

    def forward(
        self,
        h: torch.Tensor,
        *,
        start_index: Optional[int] = None,
        embedding_batch_mask: Optional[torch.Tensor] = None,
        cache_state: list[torch.Tensor],
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        assert bool(start_index is not None) ^ bool(embedding_batch_mask is not None)

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

        # Fast path to start_index based embedding lookup if available.
        # Falls back to a slower position based index lookup.
        if start_index is not None:
            xq, xk = self.embedding.forward(xq=xq, xk=xk, start_index=start_index)
        else:
            xq, xk = self.embedding.apply_batched_mask(
                xq=xq, xk=xk, mask=embedding_batch_mask
            )

        # TODO: Some model variants do some form of kv repetition to expand the
        # count of kv heads to the count of attention heads used by the q.
        assert self.head_count == self.head_count_kv, "NYI: KV expansion"

        xk_cache_update = xk
        xv_cache_update = xv

        # Tranpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        # Flash attention.
        attn_weights = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply attention mask.
        if attention_mask is not None:
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
        final_output = h + ffn_down

        # Commit the cache.
        self.cache.write(
            cache_state,
            cache_partitions=[xk_cache_update, xv_cache_update],
            transformer_block_index=self.block_index,
            page_ids=seq_block_ids,
        )
        return final_output
