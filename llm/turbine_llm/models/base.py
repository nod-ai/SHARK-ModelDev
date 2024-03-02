# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from ..data import Theta
from ..layers.core import (
    ThetaLayer,
)


class BaseCausalLMModel(ThetaLayer):
    """Base class for causal LM models.

    This provides some utilities and common API surface related to masking
    and token extraction.

    It presumes a fixed context length.
    """

    def __init__(
        self, theta: Theta, *, context_length: int, static_context_mask: bool = True
    ):
        super().__init__(theta)
        self.context_length = context_length

        if static_context_mask:
            self.register_buffer(
                "causal_context_mask", self.generate_causal_context_mask()
            )
        else:
            self.causal_context_mask = None

    def _maximally_negative_value(self, dtype):
        """Returns a maximally negative value for the given dtype.

        This can be overriden to decide on a different policy.
        """
        return float("-inf")

    def generate_causal_context_mask(self) -> torch.Tensor:
        context_length = self.context_length
        causal_context_mask = torch.triu(
            torch.ones([context_length, context_length], dtype=torch.bool),
            diagonal=1,
        )[None, None, :, :]
        return causal_context_mask

    def input_mask(
        self,
        # [bs] of integers
        seq_lens: torch.Tensor,
        batch_seqlen: int,
    ):
        """Compute a boolean input mask for a batch of sequence lengths.

        The mask will be [bs, batch_seqlen] with True at any position that is
        masked.
        """
        range_vector = torch.arange(0, batch_seqlen, 1)
        matrix = torch.unsqueeze(seq_lens, dim=-1)
        mask = range_vector >= matrix
        return mask

    def decode_attention_mask(
        self, boolean_input_mask: torch.Tensor, dtype: torch.dtype
    ):
        numeric_mask = torch.zeros_like(boolean_input_mask, dtype=dtype)
        numeric_mask.masked_fill_(
            boolean_input_mask, self._maximally_negative_value(dtype)
        )
        return numeric_mask.unsqueeze(1).unsqueeze(1)

    def attention_mask(
        self,
        input_mask: torch.Tensor,
        *,
        dtype: torch.dtype,
        causal_context_mask: Optional[torch.Tensor] = None
    ):
        """Generates a causal attention mask of [1, 1, sl, sl] of activation dtype.

        All masked positions are -inf and unmasked are 0.0.

        The pre-initialized causal context mask can be passed in. If not, then
        it will either be generated or use the initialization time buffer.
        Since this is a bool tensor of context_length^2, different deployment
        scenarios can benefit from managing this in different ways.
        """
        if causal_context_mask is None:
            causal_context_mask = self.causal_context_mask
        if causal_context_mask is None:
            causal_context_mask = self._generate_causal_context_mask()

        # Combine the causal context mask and input mask.
        _, batch_seq_len = input_mask.shape
        causal_mask = causal_context_mask[:, :, :batch_seq_len, :batch_seq_len]
        boolean_mask = causal_mask + input_mask[:, None, None, :]
        numeric_mask = torch.zeros_like(boolean_mask, dtype=dtype)
        numeric_mask.masked_fill_(boolean_mask, self._maximally_negative_value(dtype))
        return numeric_mask

    def extract_tokens_from_logits(
        self, logits: torch.Tensor, seq_lens: list[int]
    ) -> list[int]:
        """Extracts tokens from a batch of logits (B, S, D).

        The length of seq_lens must be equal to the batch size.
        Note that there are ways to code the indexing as tensor operations
        but it just creates a bunch of weirdly shaped little work on the
        accelerator. Statically looping like this is more efficient.
        """
        bs, *_ = logits.shape
        assert len(seq_lens) == bs
        results = []
        for batch, seq_len in enumerate(seq_lens):
            step_logits = logits[batch, seq_len - 1]
            results.append(torch.argmax(step_logits))
        return results
