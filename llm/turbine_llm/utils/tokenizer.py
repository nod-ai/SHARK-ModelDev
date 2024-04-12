# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Simple helpers for accessing tokenizers of various kinds."""

from abc import ABC, abstractmethod
from typing import Optional, Union

import math
import os


__all__ = [
    "load_tokenizer",
    "InferenceTokenizer",
]


class InferenceTokenizer(ABC):
    """Simple inference tokenizer."""

    def encode(
        self, texts: list[str], pad_to_multiple_of: int = 1, pad_token: int = 0
    ) -> tuple[list[list[int]]]:
        """Encodes a list of texts into a padded list of tokens.

        Returns a list of list of tokens and a list of unpadded lengths.
        """
        raw_rows = self._encode(texts)
        max_length = 0
        lengths: list[int] = []
        for row in raw_rows:
            lengths.append(len(row))
            max_length = max(max_length, len(row))
        if pad_to_multiple_of > 1:
            max_length = int(
                pad_to_multiple_of * math.ceil(max_length / pad_to_multiple_of)
            )
        for row in raw_rows:
            pad_count = max_length - len(row)
            row.extend(pad_count * [pad_token])
        return raw_rows, lengths

    def decode(self, tokens: Union[list[list[int]]], lens: Optional[list[int]] = None):
        """Decodes a list of tokens."""
        if lens is not None:
            tokens = list(tokens)
            for i, row_length in enumerate(lens):
                tokens[i] = tokens[i][0:row_length]
        return self._decode(tokens)

    @abstractmethod
    def _encode(self, texts: list[str]) -> list[list[int]]:
        ...

    @abstractmethod
    def _decode(self, tokens: list[list[int]]) -> list[str]:
        ...


def load_tokenizer(*posargs, tokenizer_type: str = "transformers", **kwargs):
    if tokenizer_type == "transformers":
        return _create_transformers_tokenizer(*posargs, **kwargs)


def _create_transformers_tokenizer(model_path: os.PathLike) -> InferenceTokenizer:
    from transformers import AutoTokenizer
    from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

    t = AutoTokenizer.from_pretrained(model_path, legacy=False)
    t.add_special_tokens({"pad_token": "0"})

    class _TransformersTokenizer(InferenceTokenizer):
        def __init__(self, t: AutoTokenizer):
            self._t = t

        def _encode(self, texts: list[str]) -> list[list[int]]:
            results = t.batch_encode_plus(
                texts,
                padding=False,
                truncation=False,
            )
            return results["input_ids"]

        def _decode(self, tokens: list[list[int]]) -> list[str]:
            return t.batch_decode(tokens)

    return _TransformersTokenizer(t)


if __name__ == "__main__":
    t = load_tokenizer("/home/stella/tmp/downloaded_open_llama_3b")
    enc, lens = t.encode(["Hi there", "who are you?"], pad_to_multiple_of=16)
    print(enc)
    print(lens)
    dec = t.decode(enc, lens)
    print(dec)
