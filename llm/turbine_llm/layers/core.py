# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn as nn

from ..data import (
    InferenceTensor,
    Theta,
)
from ..utils import debugging

__all__ = [
    "LinearLayer",
    "RotaryEmbeddingLayer",
    "RMSNormLayer",
    "ThetaLayer",
    "TokenEmbedding",
]


class BaseLayer(nn.Module):
    """Base class of all of our layers."""

    def trace_tensor(self, key: str, t: torch.Tensor, *, values: bool = True):
        debugging.trace_tensor(key, t, values=values)


class ThetaLayer(BaseLayer):
    "Base class for layers that derive parameters from a Theta object."

    def __init__(self, theta: Theta):
        super().__init__()
        self.theta = theta

    def theta_tensor(self, name: str) -> InferenceTensor:
        # TODO: We may need to do some bookkeeping here to ensure export
        # tracks all of these.
        return self.theta.tensor(name)


class LinearLayer(ThetaLayer):
    """Linear layer which computes:

    ```
    matmul(x, weight.T)
    ```

    Whether the weight is transposed as part of the calculation can be
    controlled with `transpose_weight=` (default true).
    """

    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        transpose_weight: bool = True,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.transpose_weight = transpose_weight

    def forward(self, x: torch.Tensor):
        return self.theta.ops.matmul(
            x, self.weight, transpose_rhs=self.transpose_weight
        )


class RMSNormLayer(ThetaLayer):
    """Computes the unbiased full RMS layer normalization."""

    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        epsilon: float = 1e-6,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        return self.theta.ops.rms_norm(x, self.weight, epsilon=self.epsilon)


class TokenEmbedding(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.dtype = dtype

    def forward(self, input: torch.Tensor):
        return self.theta.ops.embedding_lookup(input, self.weight, dtype=self.dtype)


class RotaryEmbeddingLayer(BaseLayer):
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

    def compute_batch_mask(
        self, start_positions: torch.Tensor, batch_seq_len: int
    ) -> torch.Tensor:
        """Computes a mask for a batch that can be repeatedly applied.

        Args:
          start_positions: Tensor of [bs] with start positions for every sequence
            in the batch.
          batch_seq_len: The sequence length dimension of the batch.
        Returns:
          Tensor of [bs, sl, 1, d] that will be later passed to apply_batch_mask.
        """
        positions_seq = torch.arange(0, batch_seq_len).unsqueeze(
            0
        ) + start_positions.unsqueeze(1)
        # Broadcast lookup to [b, ].
        freqs_cis = self._table[positions_seq]

        # Unsqueeze a unit dim for attention heads.
        broadcast_freqs_cis = freqs_cis.unsqueeze(2)
        return broadcast_freqs_cis

    def apply_batched_mask(
        self, *, xq: torch.Tensor, xk: torch.Tensor, mask: torch.Tensor
    ):
        """Applies the embedding to a ragged batch of queries and keys.

        This does a more complicated indexing operation for cases when the each
        sequence in the batch has a potentially different start position.

        positions should be of [bs, sl] and enumerate positions of all tokens.
        """
        # xq_, xk_ shape: bs, sl, _, dim
        # freqs_cis shape: max_sl, dim
        xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
        _, sl, _, dim = xq_.shape

        xq_out = torch.view_as_real(xq_ * mask).flatten(3)
        xk_out = torch.view_as_real(xk_ * mask).flatten(3)
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
