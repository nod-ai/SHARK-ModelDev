# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Manages the block cache."""

from iree.runtime import (  # type: ignore
    HalBufferView,
    HalElementType,
    BufferUsage,
    MemoryType,
)


from .config import human_size, CacheParams
from .logging import get_logger
from .session import DeviceSession, TransferBuffer, TransferBufferPool


logger = get_logger("turbine_serving.llm.cache")


class BlockCacheEntry:
    __slots__ = [
        "index",
        "in_use",
    ]

    def __init__(self, index: int):
        self.index = index
        self.in_use = False

    def __repr__(self):
        return f"Block({self.index}, {'FREE' if not self.in_use else 'BUSY'})"


class Cache:
    def __init__(self, session: DeviceSession, cache_params: CacheParams):
        self.session = session
        self.cache_params = cache_params
        self._initialize_block_cache()

    def _initialize_block_cache(self):
        model_params = self.cache_params.model
        # Allocate the on-device cache slab.
        attn_block_count = self.cache_params.device_block_count
        attn_block_size_elements = self.cache_params.attn_block_size_elements
        attn_block_size_bytes = attn_block_size_elements * model_params.attn_dtype_size
        attn_cache_size_bytes = attn_block_count * attn_block_size_bytes

        logger.info("Setting up cache for\n  %r", self.cache_params)
        logger.info(
            "Allocating attention static cache on device of %s "
            "(blocks=%s, block_size=%s bytes)",
            human_size(attn_cache_size_bytes),
            attn_block_count,
            attn_block_size_bytes,
        )
        self.attn_block_buffer = self.session.device.allocator.allocate_buffer(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=BufferUsage.DEFAULT,
            allocation_size=attn_cache_size_bytes,
        )

        # Attn block logical view.
        self.attn_block_buffer_view = HalBufferView(
            self.attn_block_buffer,
            [
                attn_block_count,
                attn_block_size_elements,
            ],
            model_params.attn_dtype,
        )

        # Accounting structs.
        self.attn_block_entries = [BlockCacheEntry(i) for i in range(attn_block_count)]
        self.attn_block_free = list(self.attn_block_entries)

    def acquire_attn_blocks(self, count: int, into_list: list[BlockCacheEntry]):
        """Acquires 'count' attention blocks.

        If there are insufficient free blocks, raises an exception.
        """
        free_list = self.attn_block_free
        assert (
            len(free_list) >= count
        ), f"Cache does not contain requested {count} free attn blocks"
        for i in range(count):
            into_list.append(free_list.pop())

    def release_attn_blocks(self, blocks: list[BlockCacheEntry]):
        """Releases a list of attention blocks.
        
        If at all possible, this should be batched to include all blocks that need to
        be released at a given time since this will trigger heavy-weight scheduling
        that will work better with a view of the new free list as a whole.
        """
        free_list = self.attn_block_free
        for block in blocks:
            free_list.append(block)
