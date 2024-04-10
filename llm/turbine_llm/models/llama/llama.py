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

from ...layers import *


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

    def __init__(self, theta: Theta, hp: configs.LlamaHParams):
        super().__init__(theta, context_length=hp.context_length)
        self.hp = hp
        # TODO: It doesn't seem like this is the right way to be getting the
        # innermost attention dim.
        attn_head_dim = self.attn_head_dim = hp.rope_dimension_count
        self.cache = PagedKVCache(
            transformer_block_count=hp.block_count,
            attn_head_count=hp.attention_head_count_kv,
            attn_head_dim=attn_head_dim,
            cache_partition_count=2,
            block_seq_stride=16,
        )
        self.add_module(
            "token_embedding",
            TokenEmbeddingLayer(theta("token_embd"), dtype=hp.activation_dtype),
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
            if block_idx == 0:
                self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
            h = block(
                h,
                embedding=self.attention_embedding,
                start_index=0,
                attention_mask=attention_mask,
                write_cache_state=cache_state,
                seq_block_ids=seq_block_ids,
            )
            self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)
        return logits

    def decode(
        self,
        # [bs, 1]
        tokens: torch.Tensor,
        *,
        # [bs, 1, 1, batch_seq_len]
        attention_mask: torch.Tensor,
        # [bs] of starting positions
        start_positions: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        read_cache_state: list[torch.Tensor],
        write_cache_state: list[torch.Tensor],
    ):
        bs, _ = tokens.shape
        # Precompute a position based mask for computing rope embeddings
        # as it is the same for all blocks.
        embedding_batch_mask = self.attention_embedding.compute_batch_mask(
            start_positions, batch_seq_len=1
        )
        self.trace_tensor("llama.embedding_batch_mask", embedding_batch_mask)

        # Allocate per-block temporary K/V tensors. These temporaries hold
        # one block's K/V state for the maximum context length.
        xk_temp = torch.empty(
            [
                bs,
                self.context_length,
                self.hp.attention_head_count_kv,
                self.attn_head_dim,
            ],
            dtype=self.hp.activation_dtype,
        )
        xv_temp = torch.empty(
            [
                bs,
                self.context_length,
                self.hp.attention_head_count_kv,
                self.attn_head_dim,
            ],
            dtype=self.hp.activation_dtype,
        )

        h = self.token_embedding(tokens)
        self.trace_tensor("llama.token_embedding", h)

        # Iterate over attention blocks.
        for block_idx, block in enumerate(self.attn_blocks):
            if block_idx == 0:
                self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
            h = block(
                h,
                start_positions=start_positions,
                embedding=self.attention_embedding,
                embedding_batch_mask=embedding_batch_mask,
                attention_mask=attention_mask,
                read_cache_state=read_cache_state,
                write_cache_state=write_cache_state,
                seq_block_ids=seq_block_ids,
                xk_temp=xk_temp,
                xv_temp=xv_temp,
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
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv

    def forward(
        self,
        h: torch.Tensor,
        *,
        embedding: RotaryEmbeddingLayer,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        start_index: Optional[int] = None,
        start_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embedding_batch_mask: Optional[torch.Tensor] = None,
        write_cache_state: Optional[list[torch.Tensor]] = None,
        read_cache_state: Optional[list[torch.Tensor]] = None,
        xk_temp: Optional[torch.Tensor] = None,
        xv_temp: Optional[torch.Tensor] = None,
    ):
        assert bool(start_index is not None) ^ bool(embedding_batch_mask is not None)

        x = self.attn_norm(h)

        bs, batch_seq_len, feature_dim = x.shape
        assert feature_dim == self.head_count * self.head_dim

        xq = self.attn_q(x)
        xk = self.attn_k(x)
        xv = self.attn_v(x)

        xq = xq.view(bs, batch_seq_len, self.head_count, self.head_dim)
        xk = xk.view(bs, batch_seq_len, self.head_count_kv, self.head_dim)
        xv = xv.view(bs, batch_seq_len, self.head_count_kv, self.head_dim)

        # Fast path to start_index based embedding lookup if available.
        # Falls back to a slower position based index lookup.
        if start_index is not None:
            xq, xk = embedding.forward(xq=xq, xk=xk, start_index=start_index)
        else:
            xq, xk = embedding.apply_batched_mask(
                xq=xq, xk=xk, mask=embedding_batch_mask
            )

        # TODO: Some model variants do some form of kv repetition to expand the
        # count of kv heads to the count of attention heads used by the q.
        assert self.head_count == self.head_count_kv, "NYI: KV expansion"

        xk_cache_update = xk
        xv_cache_update = xv

        # Manage the cache.
        if read_cache_state is None:
            # If not instructed to read from the cache, we assume this is
            # prefill and the projected K/V values represent the complete
            # sequence.
            # Commit the whole cache if writing is enabled.
            if write_cache_state is not None:
                # TODO: Do a full pages write or a single row write, depending
                # on whether prefill vs decode.
                if read_cache_state is None:
                    # Overwrite the whole cache.
                    self.cache.write(
                        write_cache_state,
                        cache_partitions=[xk_cache_update, xv_cache_update],
                        transformer_block_index=self.block_index,
                        page_ids=seq_block_ids,
                    )
        else:
            # We need to initialize/read the K/V from the cache for the whole
            # sequence. Note that at this point, it is possible to fork and
            # use a memory efficient attention kernel that can do indirect
            # reads, skipping this materialization. This path is taken for
            # a decode step.
            assert start_positions is not None
            assert xk_temp is not None and xv_temp is not None

            kv_seq_len = seq_block_ids.shape[1] * self.cache.block_seq_stride

            if write_cache_state:
                # Write our one updated cache row into the cache.
                self.cache.write_timestep(
                    write_cache_state,
                    cache_partitions=[
                        xk_cache_update,
                        xv_cache_update,
                    ],
                    transformer_block_index=self.block_index,
                    seq_positions=start_positions + 1,
                    page_ids=seq_block_ids,
                )

            # Restore from the cache.
            self.cache.read(
                read_cache_state,
                read_into_partitions=[
                    xk_temp[:, 0:kv_seq_len, ...],
                    xv_temp[:, 0:kv_seq_len, ...],
                ],
                transformer_block_index=self.block_index,
                page_ids=seq_block_ids,
            )

            # For computation, we create a subview of the xk/xv tensors to have
            # a sequence length covering the blocked size. This must include
            # the newly added row (the caller is responsible for ensuring that
            # every block has at least one row left). We'll compute on this
            # ragged view and use an appropriate mask.
            xk = xk_temp[:, 0:kv_seq_len, ...]
            xv = xv_temp[:, 0:kv_seq_len, ...]

        # Tranpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        # Flash attention.
        attn_weights = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        self.assert_not_nan(attn_weights)

        # Apply attention mask.
        self.trace_tensor("attn_weights", attn_weights, values=False)
        if attention_mask is not None:
            # self.trace_tensor("attn_mask", attention_mask)
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(xq)
        attn_output = torch.matmul(attn_weights, values)  # (bs, heads, slen, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bs, batch_seq_len, -1)

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

        return final_output
