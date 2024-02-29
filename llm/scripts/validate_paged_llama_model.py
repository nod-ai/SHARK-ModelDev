# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

from turbine_llm.config import *
from turbine_llm.data import *
from turbine_llm.models.llama import *


def main(args: list[str]):
    torch.no_grad().__enter__()
    config = load_gguf_file(args[0])
    hp = LlamaHParams.from_gguf_props(config.properties)
    model = PagedLlamaModelV1(config.root_theta, hp)
    cache_state = model.cache.allocate(128, torch.float32)
    start_index = 0
    next_batch = torch.tensor(
        [
            [
                1,
                1059,
                31871,
                1217,
                322,
                266,
                3682,
                6075,
                31902,
                13,
                31849,
                31871,
                0,
                0,
                0,
                0,
            ]
            + 48 * [0],
            [
                1,
                1059,
                31871,
                1217,
                322,
                31871,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            + 48 * [0],
            64 * [0],
            64 * [0],
        ]
    )
    assert next_batch.shape[1] % model.cache.block_seq_stride == 0
    seq_block_ids = torch.tensor(
        [
            [127, 0, 0, 0],
            [126, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    seq_lens = [12, 6, 0, 0]

    attention_mask = model.attention_mask(
        model.input_mask(torch.tensor(seq_lens), next_batch.shape[1])
    )

    print(f"Step {start_index}")
    logits = model.prefill(
        next_batch,
        attention_mask=attention_mask,
        seq_block_ids=seq_block_ids,
        cache_state=cache_state,
    )
    tokens = model.extract_tokens_from_logits(logits, seq_lens)
    print(f"  : tokens = {tokens}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
