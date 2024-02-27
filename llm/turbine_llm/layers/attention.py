# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import Theta
from .core import (
    LinearLayer,
    RMSNormLayer,
    ThetaLayer,
)

__all__ = [
    "LlamaAttentionBlock",
    "RotaryEmbeddingLayer",
]


class RotaryEmbeddingLayer(nn.Module):
    """Computes a rotary embedding in the style popularized by llama (RoPE)."""

    def __init__(self, *, rope_dimension_count: int, max_seqlen: int):
        super().__init__()
        self._table = self._create_rotary_embed_table(
            max_seqlen=max_seqlen, dim=rope_dimension_count
        )

    def forward(self, *, xq: torch.Tensor, xk: torch.Tensor, start_index: int):
        # xq_, xk_ shape: bs, sl, _, dim
        # freqs_cis shape: max_sl, dim
        xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
        _, sl, _, dim = xq_.shape

        # Offset the table based on starting position.
        freqs_cis = self._table[start_index:sl, :]
        assert freqs_cis.shape[-1] == dim
        assert freqs_cis.shape[0] >= sl, "Sequence length longer than embedding table"

        broadcast_freqs_cis = freqs_cis[None, 0:sl, None, :]
        xq_out = torch.view_as_real(xq_ * broadcast_freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * broadcast_freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    @staticmethod
    def _create_rotary_embed_table(
        max_seqlen: int, dim: int, theta_value: float = 10000.0
    ):
        freqs = 1.0 / (
            theta_value ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        t = torch.arange(max_seqlen, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


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

        xq, xk = self.embedding(xq, xk, start_index=start_index)

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
        ff_input = self.ffn_norm(h)
        ff_gate = F.silu(self.ff_gate(ff_input))
        ff_up = self.ff_up(ff_input)
        ff_down = self.ff_down(ff_gate * ff_up)
        h = h + ff_down

        return h
