# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implements the BatchGenerateService for V1 compiled models.

This is far from where we want to land but is intended for first round bootstrapping.
Perhaps the biggest issue is that it wouldn't mate well as-is with samplers.
"""

from dataclasses import dataclass

import numpy as np

from iree.runtime import (  # type: ignore
    HalCommandBuffer,
    HalElementType,
    VmFunction,
    VmVariantList,
)

from ..cache import BlockCacheEntry, Cache
from ..config import ServiceParams
from ..logging import get_logger, NDEBUG
from ..service import BatchGenerateRequest, BatchGenerateService
from ..session import (
    AsyncResources,
    DeviceSession,
    HostContext,
    TransferBufferPool,
    WorkQueue,
)

logger = get_logger("turbine_serving.llm.impl.service_v1")

EXPECTED_CONCURRENCY = 10


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
        max_bs = params.model.max_batch_size
        max_sl = params.model.max_seq_len
        initial_inflight = EXPECTED_CONCURRENCY

        # block_indices_pool: array([max_batch_size, max_attn_blocks], np.int16)
        # Suitable to handle the sequence->block mapping for all steps.
        self.block_indices_pool = TransferBufferPool.shaped(
            self.session,
            [
                max_bs,
                max_sl // self.block_pos_stride,
            ],
            HalElementType.INT_16,
            initial_capacity=initial_inflight,
            growable=True,
            name="block_cache_indices",
        )

        # Prefill tokens: array([max_batch_size, max_seq_len], np.int32)
        # Tokens inputs to prefill.
        self.prefill_tokens_pool = TransferBufferPool.shaped(
            self.session,
            [
                max_bs,
                max_sl,
            ],
            HalElementType.INT_32,
            initial_capacity=initial_inflight,
            growable=True,
            name="prefill_tokens",
        )

        # Prefill sequence lengths: array([max_batch_size], np.int16)
        # Sequence lengths of input tokens.
        self.prefill_seq_lens_pool = TransferBufferPool.shaped(
            self.session,
            [max_bs],
            HalElementType.INT_16,
            initial_capacity=initial_inflight,
            growable=True,
            name="prefill_seq_lens",
        )

    def start_prefill(self, batch_request: BatchGenerateRequest):
        block_pos_stride = self.block_pos_stride
        cache = self.cache

        # Loop through each request and reserve initial attention blocks.
        bs = 0
        sequences: list[_Sequence] = []
        max_attn_blocks_length = 0
        max_seq_length = 0

        for req in batch_request.requests:
            bs += 1
            seq = _Sequence()
            sequences.append(seq)
            seq.prefill_token_ids = prefill_token_ids = req.required_prompt_token_ids
            seq_length = len(prefill_token_ids)
            seq.seq_length = seq_length
            max_seq_length = max(max_seq_length, seq_length)
            initial_block_count = seq_length // block_pos_stride + 1
            logger.debug("Acquire prefill attn blocks: %s", initial_block_count)
            cache.acquire_attn_blocks(initial_block_count, seq.attn_blocks)
            max_attn_blocks_length = max(max_attn_blocks_length, initial_block_count)

        # Determine the appropriate batched entrypoints.
        for allowed_bs in self.batch_sizes:
            if allowed_bs >= bs:
                prefill_function = self.prefill_functions[allowed_bs]
                decode_function = self.decode_functions[allowed_bs]
                break
        else:
            raise AssertionError(f"Unsupported batch size: {bs}")

        # Initialize the state machine.
        hc = self.module_set.host_context
        state = _State(
            bs=allowed_bs,
            queue=hc.session.queue(),
            sequences=sequences,
            max_attn_blocks_length=max_attn_blocks_length,
            max_seq_length=max_seq_length,
            prefill_function=prefill_function,
            decode_function=decode_function,
        )

        # Schedule invocation work (on a dedicated host context/thread).
        hc.loop.call_soon_threadsafe(lambda: self._invoke_prefill(hc, state))

    def _invoke_prefill(self, hc: HostContext, state: "_State"):
        # Record a command buffer for performing h2d transfers.
        cb = HalCommandBuffer(hc.session.device)

        resources = state.resources
        bs = state.bs
        max_seq_length = state.max_seq_length
        max_attn_blocks_length = state.max_attn_blocks_length

        # Prepare input tokens, sequence lengths and block indices.
        # We acquire a transfer buffer of each from the respective pool, populate its
        # host side and enqueue.
        # prefill_tokens: array([bs, max_seq_length], np.int32)
        prefill_tokens_host, prefill_tokens_device = resources.acquire_transfer_buffer(
            self.prefill_tokens_pool
        ).h2d_array(cb, [bs, max_seq_length], HalElementType.INT_32, fill_value=0)

        # prefill_seq_lens: array([bs], np.int32)
        (
            prefill_seq_lens_host,
            prefill_seq_lens_device,
        ) = resources.acquire_transfer_buffer(self.prefill_seq_lens_pool).h2d_array(
            cb, [bs], HalElementType.INT_32, fill_value=0
        )

        # attn_block_indices: array([bs, max_attn_blocks], np.in16)
        (
            prefill_attn_block_indices_host,
            prefill_attn_block_indices_device,
        ) = resources.acquire_transfer_buffer(self.block_indices_pool).h2d_array(
            cb, [bs, max_attn_blocks_length], HalElementType.INT_16, fill_value=0
        )

        # Populate host buffers for each sequence.
        for i in range(len(state.sequences)):
            seq = state.sequences[i]
            attn_blocks = seq.attn_blocks
            prefill_token_ids = seq.prefill_token_ids
            row_seq_len = len(prefill_token_ids)
            prefill_tokens_host[i, 0:row_seq_len] = prefill_token_ids
            prefill_seq_lens_host[i] = row_seq_len
            for j in range(len(seq.attn_blocks)):
                prefill_attn_block_indices_host[i, j] = attn_blocks[j].index

        # Perform h2d transfers.
        cb.end()
        state.queue.execute_sequential([cb])

        # Inputs:
        #   token_ids
        #   seq_lens
        #   attn_block_indices
        #   attn_block_buffer_view (the entire slab passed as input)
        #   wait, signal semaphores
        #   tied attn_block_buffer (for input[2])
        #   tied attn_block_buffer (for result[0])
        attn_block_buffer_view = self.cache.attn_block_buffer_view
        attn_block_buffer = self.cache.attn_block_buffer
        inputs = VmVariantList(3)
        inputs.push_ref(prefill_tokens_device)
        inputs.push_ref(prefill_seq_lens_device)
        inputs.push_ref(prefill_attn_block_indices_device)
        inputs.push_ref(attn_block_buffer_view)
        inputs.push_ref(attn_block_buffer)  # Tied input[3]
        inputs.push_ref(attn_block_buffer)  # Tied result[0]
        wait_fence, signal_fence = state.queue.step_fences()
        inputs.push_ref(wait_fence)
        inputs.push_ref(signal_fence)

        # Outputs:
        #   attn_block_buffer_view (tied output)
        #   decode_tokens
        outputs = VmVariantList(1)
        hc.vm_context.invoke(state.prefill_function, inputs, outputs)

        # TODO: Do not wait on signal fence.
        signal_fence.wait()
        print("INVOKE prefill output:", outputs)


class _Sequence:
    __slots__ = [
        "attn_blocks",
        "prefill_token_ids",
        "seq_length",
    ]

    prefill_token_ids: list[int]

    def __init__(self):
        self.seq_length: int = 0
        self.attn_blocks: list[BlockCacheEntry] = []


class _State:
    __slots__ = [
        "bs",
        "queue",
        "decode_function",
        "max_attn_blocks_length",
        "max_seq_length",
        "prefill_function",
        "resources",
        "sequences",
    ]

    def __init__(
        self,
        bs: int,
        queue: WorkQueue,
        max_seq_length: int,
        max_attn_blocks_length: int,
        sequences: list[_Sequence],
        prefill_function: VmFunction,
        decode_function: VmFunction,
    ):
        self.resources = AsyncResources()
        self.bs = bs
        self.queue = queue
        self.max_seq_length = max_seq_length
        self.max_attn_blocks_length = max_attn_blocks_length
        self.sequences = sequences
        self.prefill_function = prefill_function
        self.decode_function = decode_function
