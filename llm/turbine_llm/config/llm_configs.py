# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structured configuration objects for various LLMs.

This draws heavily from the work that ggml has done to systematize the state
of the world for GGUF files:
  https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

When in question, we draw from the vocabulary and normalization they have done
(and indeed, can bootstrap these off of GGUF files).
"""

from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "LlamaHParams",
]


@dataclass
class LlamaHParams:
    """Corresponds 1:1 with the 'LLM' section of the GGUF docs.

    Comments are only provided if they differ from this source.
    """

    context_length: int
    embedding_length: int
    block_count: int
    feed_forward_length: int
    rope_dimension_count: int
    attention_head_count: int
    attention_layer_norm_rms_epsilon: float
    attention_head_count_kv: int

    @staticmethod
    def from_gguf_props(p: dict[str, Any]):
        attention_head_count = _int_prop(p, "llama.attention.head_count")
        return LlamaHParams(
            context_length=_int_prop(p, "llama.context_length"),
            embedding_length=_int_prop(p, "llama.embedding_length"),
            block_count=_int_prop(p, "llama.block_count"),
            feed_forward_length=_int_prop(p, "llama.feed_forward_length"),
            rope_dimension_count=_int_prop(p, "llama.rope.dimension_count"),
            attention_head_count=attention_head_count,
            attention_layer_norm_rms_epsilon=_float_prop(
                p, "llama.attention.layer_norm_rms_epsilon"
            ),
            attention_head_count_kv=_optional_int_prop(
                p, "llama.attention.head_count_kv", attention_head_count
            ),
        )


def _float_prop(p: dict[str, Any], name: str) -> float:
    try:
        return float(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be a float and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def _int_prop(p: dict[str, Any], name: str) -> int:
    try:
        return int(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def _optional_int_prop(p: dict[str, Any], name: str, default_value: int) -> int:
    value = p[name]
    if value is None:
        return default_value
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e
