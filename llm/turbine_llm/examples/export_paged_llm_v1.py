# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inference support for the PagedLLMV1 protocol of models."""

import torch

from shark_turbine.aot import *

from turbine_llm.layers import *
from turbine_llm.types import *

# TODO: Should be using a base class with the protocol supported.
from ..models.llama.llama import LlamaModelConfig, PagedLlamaModelV1


def main():
    from ..utils import cli

    parser = cli.create_parser()
    cli.add_gguf_dataset_options(parser)
    args = cli.parse(parser)

    data_files = cli.get_gguf_data_files(args)
    dataset = gguf_interop.load_file(data_files["gguf"])

    hp = configs.LlamaHParams.from_gguf_props(dataset.properties)
    model = PagedLlamaModelV1(dataset.root_theta, LlamaModelConfig(hp))

    # Unrolling cache updates by batch row makes dynamo sad without an
    # override. There may be a better way to do this.
    import torch._dynamo.config as dynamo_config

    # TODO: Seems removed from 2.3+
    # dynamo_config.max_loop_unroll_nodes = 0

    fxb = FxProgramsBuilder(model)

    def generate_batch_prefill(bs: int):
        tokens = torch.empty(bs, 64, dtype=torch.int64)
        seq_lens = torch.empty(bs, dtype=torch.int64)
        seq_block_ids = torch.empty(bs, 4, dtype=torch.int64)
        cache_state = model.cache.allocate(128, torch.float32)
        block_dim = torch.export.Dim("block", max=2047 // 16)
        sl_dim = 16 * block_dim
        page_dim = torch.export.Dim("page")
        dynamic_shapes = {
            "tokens": {1: sl_dim},
            "seq_lens": {},
            "seq_block_ids": {1: block_dim},
            "cache_state": [{0: page_dim}],
        }

        print(f"Exporting prefill_bs{bs}")

        @fxb.export_program(
            name=f"prefill_bs{bs}",
            args=(tokens, seq_lens, seq_block_ids, cache_state),
            dynamic_shapes=dynamic_shapes,
        )
        def _(model, tokens, seq_lens, seq_block_ids, cache_state):
            sl = tokens.shape[1]
            input_mask = model.input_mask(seq_lens, sl)
            attention_mask = model.attention_mask(input_mask, dtype=torch.float32)
            logits = model.prefill(
                tokens,
                attention_mask=attention_mask,
                seq_block_ids=seq_block_ids,
                cache_state=cache_state,
            )
            return logits

    def generate_batch_decode(bs: int):
        tokens = torch.ones(bs, 1, dtype=torch.int64)
        seq_lens = torch.ones(bs, dtype=torch.int64)
        start_positions = torch.ones(bs, dtype=torch.int64)
        seq_block_ids = torch.zeros(bs, 4, dtype=torch.int64)
        cache_state = model.cache.allocate(128, torch.float32)
        block_dim = torch.export.Dim("block", max=2047 // 16)
        page_dim = torch.export.Dim("page")
        dynamic_shapes = {
            "tokens": {},
            "seq_lens": {},
            "start_positions": {},
            "seq_block_ids": {1: block_dim},
            "cache_state": [{0: page_dim}],
        }

        print(f"Exporting decode_bs{bs}")

        @fxb.export_program(
            name=f"decode_bs{bs}",
            args=(
                tokens,
                seq_lens,
                start_positions,
                seq_block_ids,
                cache_state,
            ),
            dynamic_shapes=dynamic_shapes,
        )
        def _(
            model,
            tokens,
            seq_lens,
            start_positions,
            seq_block_ids,
            cache_state,
        ):
            input_mask = model.input_mask(
                seq_lens, seq_block_ids.shape[1] * model.cache.block_seq_stride
            )
            attention_mask = model.decode_attention_mask(
                input_mask, dtype=torch.float32
            )
            logits = model.decode(
                tokens,
                attention_mask=attention_mask,
                start_positions=start_positions,
                seq_block_ids=seq_block_ids,
                cache_state=cache_state,
            )
            return logits

    generate_batch_prefill(4)
    generate_batch_decode(4)
    print("GENERATED!")

    for name, ep in fxb.programs.items():
        print(f"EXPORT {name}:\n{ep}")

    print("Exporting")
    output = export(fxb)
    print("Saving")
    output.save_mlir("/tmp/batch_llama_v1.mlir")


if __name__ == "__main__":
    main()
