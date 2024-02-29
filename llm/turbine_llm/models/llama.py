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
    PagedLlamaAttentionBlock,
    RotaryEmbeddingLayer,
)

from ..layers.kv_cache import (
    PagedKVCache,
)

from ..layers.core import (
    LinearLayer,
    RMSNormLayer,
    ThetaLayer,
    TokenEmbedding,
)

from .base import BaseCausalLMModel


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
            )
            self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)
        return logits
