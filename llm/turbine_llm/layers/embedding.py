# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn as nn


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
