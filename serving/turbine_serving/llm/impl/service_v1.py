# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implements the BatchGenerateService for V1 compiled models."""

from dataclasses import dataclass

from iree.runtime import (  # type: ignore
    HalElementType,
    VmFunction,
    VmVariantList,
)

from ..cache import BlockCacheEntry, Cache
from ..config import ServiceParams
from ..logging import get_logger
from ..service import BatchGenerateRequest, BatchGenerateService
from ..session import DeviceSession, HostContext, TransferBuffer, TransferBufferPool

logger = get_logger("turbine_serving.llm.impl.service_v1")


class GenerateServiceV1(BatchGenerateService):
    def __init__(self, session: DeviceSession, params: ServiceParams):
        self.params = params
        self.block_pos_stride = params.cache.block_pos_stride
        self.batch_sizes = params.model.prefill_batch_sizes
        # TODO: Remove distinction between prefill and decode batch sizes.
        assert params.model.decode_batch_sizes == self.batch_sizes
        self.session = session
        self.cache = Cache(session, params.cache)
        module_name = params.model.module_name
        logger.info("Configuring serving for module set %s", module_name)
        self.module_set = session.module_set(params.model.module_name)

        # Initialize prefill entry-points (1 per batch size).
        self.prefill_functions: dict[int, VmFunction] = {}
        for bs in self.batch_sizes:
            assert bs not in self.prefill_functions
            symbol_name = f"prefill_bs{bs}"
            logger.info("Looking up symbol '%s'", symbol_name)
            self.prefill_functions[bs] = self.module_set.function(
                module_name, symbol_name
            )

        # Initialize decode entry-points (1 per batch size).
        self.decode_functions: dict[int, VmFunction] = {}
        for bs in self.batch_sizes:
            assert bs not in self.decode_functions
            symbol_name = f"decode_bs{bs}"
            logger.info("Looking up symbol '%s'", symbol_name)
            self.decode_functions[bs] = self.module_set.function(
                module_name, symbol_name
            )

        self._initialize_transfer_pools()

    def _initialize_transfer_pools(self):
        params = self.params
        initial_inflight = 10
        # Block indices of shape [max_batch_size, max_attn_blocks]
        self.block_indices_pool = TransferBufferPool.shaped(
            self.session,
            [
                params.model.max_batch_size,
                params.model.max_seq_len // self.block_pos_stride,
            ],
            HalElementType.INT_16,
            initial_capacity=initial_inflight,
            growable=True,
            name="block_cache_indices",
        )

    def start_prefill(self, batch_request: BatchGenerateRequest):
        block_pos_stride = self.block_pos_stride
        cache = self.cache

        # Loop through each request and reserve initial attention blocks.
        bs = 0
        sequences: list[_Sequence] = []
        max_seq_length = 0

        for req in batch_request.requests:
            bs += 1
            seq = _Sequence()
            sequences.append(seq)
            seq_length = len(req.prompt_token_ids)
            seq.seq_length = seq_length
            max_seq_length = max(max_seq_length, seq_length)
            initial_block_count = seq_length // block_pos_stride + 1
            logger.debug("Acquire prefill attn blocks: %s", initial_block_count)
            cache.acquire_attn_blocks(initial_block_count, seq.attn_blocks)

        # Determine the appropriate batched entrypoints.
        for allowed_bs in self.batch_sizes:
            if allowed_bs >= bs:
                prefill_function = self.prefill_functions[allowed_bs]
                decode_function = self.decode_functions[allowed_bs]
                break
        else:
            raise AssertionError(f"Unsupported batch size: {bs}")

        # Initialize the state machine.
        state = _State(
            block_indices=self.block_indices_pool.acquire(),
            sequences=sequences,
            max_seq_length=max_seq_length,
            prefill_function=prefill_function,
            decode_function=decode_function,
        )

        # Schedule work.
        hc = self.module_set.context
        hc.schedule(lambda: self._invoke_prefill(hc, state))

    def _invoke_prefill(self, hc: HostContext, state: "_State"):
        attn_block_buffer_view = self.cache.attn_block_buffer_view
        attn_block_buffer = self.cache.attn_block_buffer

        # Inputs:
        #   token_ids
        #   seq_lens
        #   attn_block_buffer_view (the entire slab passed as input)
        #   wait, signal semaphores
        #   tied attn_block_buffer (for input[2])
        #   tied attn_block_buffer (for result[0])
        inputs = VmVariantList(3)
        # TODO: Token ids
        # TODO: Seq lens
        inputs.push_ref(attn_block_buffer_view)
        # TODO: Semaphores.
        inputs.push_ref(attn_block_buffer)  # Tied input[1]
        inputs.push_ref(attn_block_buffer)  # Tied result[0]

        # Outputs:
        #   attn_block_buffer_view (tied output)
        #   decode_tokens
        outputs = VmVariantList(1)
        hc.vm_context.invoke(state.prefill_function, inputs, outputs)
        print("INVOKE prefill output:", outputs)


class _Sequence:
    __slots__ = [
        "attn_blocks",
        "seq_length",
    ]

    def __init__(self):
        self.seq_length: int = 0
        self.attn_blocks: list[BlockCacheEntry] = []


class _State:
    __slots__ = [
        "block_indices",
        "decode_function",
        "max_seq_length",
        "prefill_function",
        "sequences",
    ]

    def __init__(
        self,
        block_indices: TransferBuffer,
        max_seq_length: int,
        sequences: list[_Sequence],
        prefill_function: VmFunction,
        decode_function: VmFunction,
    ):
        self.block_indices = block_indices
        self.max_seq_length = max_seq_length
        self.sequences = sequences
        self.prefill_function = prefill_function
        self.decode_function = decode_function
