# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Light-weight encapsulations for various forms of attention KV-caches.

These are not complete abstractions: they are primarily focused on making
tightly coupled transformer blocks a bit less "stringy" with loose tensors
and dims floating around everywhere. 
"""

import math

import torch

from ..utils.debugging import trace_tensor

__all__ = [
    "PagedKVCache",
]


class PagedKVCache:
    """Implementation of a KV cache on top of a 'page table'.

    The page table slab is physically represented as a 2D tensor:
        [page_count, flattened_dims]

    Each "page" can be thought of as a 6D view onto:

    * transformer block
    * cache partition (K or V cache)
    * block sequence stride (number of sequence positions per block)
    * attention heads
    * attention dimensionality

    Note that the internal page structure matches the organization of the
    model, allowing contiguous individual local reads and writes at a sub-block
    granularity if indexing deeply into the structure.
    """

    def __init__(
        self,
        *,
        transformer_block_count: int,
        attn_head_count: int,
        attn_head_dim: int,
        cache_partition_count: int = 2,
        block_seq_stride: int = 16,
    ):
        self.transformer_block_count = transformer_block_count
        self.attn_head_count = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.cache_partition_count = cache_partition_count
        self.block_seq_stride = block_seq_stride

        # Some derived values based on attributes.
        self.sub_page_dims = [
            self.transformer_block_count,
            self.cache_partition_count,
            self.block_seq_stride,
            self.attn_head_count,
            self.attn_head_dim,
        ]
        self.page_slab_flat_dim = math.prod(self.sub_page_dims)

    def unflatten_page_table(self, state: list[torch.Tensor]) -> torch.Tensor:
        """Unflattens the 2D page table to a 6D tensor."""
        assert len(state) == 1, f"Expected 1-element state. Got: {len(state)}"
        page_slab = state[0]
        return page_slab.reshape(
            [
                -1,
                self.transformer_block_count,
                self.cache_partition_count,
                self.block_seq_stride,
                self.attn_head_count,
                self.attn_head_dim,
            ]
        )

    def allocate(self, page_count: int, dtype: torch.dtype) -> list[torch.Tensor]:
        """Allocates tensor state for a page table for the given capacity in
        pages.
        """
        return [torch.empty([page_count, self.page_slab_flat_dim], dtype=dtype)]

    def read(
        self,
        state: list[torch.Tensor],
        *,
        read_into_partitions: list[torch.Tensor],
        transformer_block_index: int,
        page_ids: torch.Tensor,
    ):
        """Reads cache partitions from the page table for the given page_ids.

        Args:
        state: State struct as returned from allocate().
        read_into_partitions: List of cache partitions to read into in-place.
        transformer_block_index: The index of the transformer block accessing
            the cache.
        page_ids: Tensor of [bs, max_seqlen // block_pos_stride] of page ids
            to access.

        Returns a tuple of cache partitions (i.e. k and v caches for the transformer
        block), linearized. Note that this reference approach to reading by
        materializing linearly may not be terribly efficient unless if the
        compiler can fuse the gather.
        """
        page_table = self.unflatten_page_table(state)  # 6D

        bs, block_seq_len, *_ = page_ids.shape
        # Blocks dim 1,2 according to the configured block stride.
        blocked_shape = [
            bs,
            block_seq_len,
            self.block_seq_stride,
            self.attn_head_count,
            self.attn_head_dim,
        ]

        # Reshape the page cache into sub-blocks so that we can index at the
        # granularity of the transformer_block and cache partition.
        # This requires us to recompute indices to the sub-block reference
        # frame.
        # The subblock slab is organized as:
        #   [page, attn_layer, cache_partition]
        # Where the cache line can be 0 (k) or 1 (v).
        subblock_table = page_table.flatten(start_dim=0, end_dim=2)
        page_stride = self.transformer_block_count * self.cache_partition_count
        transformer_block_stride = self.cache_partition_count
        base_subblock_ids = page_ids * page_stride + (
            transformer_block_index * transformer_block_stride
        )

        def read_cache_partition(index: int, into_partition: torch.Tensor):
            subblock_ids = (
                (base_subblock_ids + index) if index > 0 else base_subblock_ids
            )
            # TODO: Potentially clamp all page 0 indices to the mask value.
            # Or even better, require that the ids are replicated such that access is
            # legal.
            # Now for each of the k/v attn_block_ids, which have been adjusted to
            # index into the sub-pages, we flatten to do a linear index_select
            # copy of the sub-blocks by collapsing the first two dims so we have
            # a linear list.
            # TODO: Can be rewritten into inplace with out= on index_select.
            selected = (
                torch.index_select(subblock_table, 0, subblock_ids.flatten(0, 1))
                .unflatten(0, blocked_shape[0:2])
                .flatten(1, 2)
            )
            # trace_tensor("kv.selected", selected)
            into_partition[...] = selected

        for index, read_into_partition in enumerate(read_into_partitions):
            read_cache_partition(index, read_into_partition)

    def write_timestep(
        self,
        state: list[torch.Tensor],
        # List of [bs, 1, attn_head_count, attn_head_dim]
        cache_partitions: list[torch.Tensor],
        *,
        transformer_block_index: int,
        # [bs]
        seq_positions: torch.Tensor,
        # [bs, max_seqlen // block_pos_stride]
        page_ids: torch.Tensor,
    ):
        """Writes a single batched timestep across all cache partitions.

        Note that this internally loops over the batch size, which cannot be
        dynamic.
        """
        # [
        #     -1,
        #     self.transformer_block_count,
        #     self.cache_partition_count,
        #     self.block_seq_stride,
        #     self.attn_head_count,
        #     self.attn_head_dim,
        # ]
        page_table = self.unflatten_page_table(state)  # 6D
        bs, *_ = seq_positions.shape
        assert len(cache_partitions) == self.cache_partition_count
        for i in range(bs):
            position = seq_positions[i]
            page_id = page_ids[i, position // self.block_seq_stride]
            page_offset = position % self.block_seq_stride
            for partition_index in range(self.cache_partition_count):
                cache_partition = cache_partitions[partition_index]
                print(
                    f"WRITE TIMESTEP: {(page_id, transformer_block_index, partition_index, page_offset)}"
                )
                page = page_table[
                    page_id, transformer_block_index, partition_index, page_offset
                ]

                def dump_page(banner):
                    print(banner)
                    for j in range(self.block_seq_stride):
                        print(
                            f"  [{j}] = {page_table[page_id, transformer_block_index, partition_index, j]}"
                        )

                # dump_page("PAGE BEFORE:")
                page[...] = cache_partition[i, 0]
                # dump_page("PAGE BEFORE:")

    def write(
        self,
        state: list[torch.Tensor],
        cache_partitions: list[torch.Tensor],
        *,
        transformer_block_index: int,
        page_ids: torch.Tensor,
    ):
        """Writes cache partitions from a linear layout to the page table.

        This is the inverse of the linear read. The same caveat applies if the
        in-place scatter cannot be fused.
        """
        page_table = self.unflatten_page_table(state)  # 6D

        bs, block_seq_len, *_ = page_ids.shape
        # Blocks dim 1,2 according to the configured block stride.
        blocked_shape = [
            bs,
            block_seq_len,
            self.block_seq_stride,
            self.attn_head_count,
            self.attn_head_dim,
        ]

        # Reshape the page cache into sub-blocks so that we can index at the
        # granularity of the transformer_block and cache partition.
        # This requires us to recompute indices to the sub-block reference
        # frame.
        # The subblock slab is organized as:
        #   [page, attn_layer, cache_partition]
        # Where the cache line can be 0 (k) or 1 (v).
        subblock_table = page_table.flatten(start_dim=0, end_dim=2)
        page_stride = self.transformer_block_count * self.cache_partition_count
        transformer_block_stride = self.cache_partition_count
        base_subblock_ids = page_ids * page_stride + (
            transformer_block_index * transformer_block_stride
        )

        def write_cache_partition(index: int, part: torch.Tensor):
            part_block_view = part.reshape(blocked_shape)
            subblock_ids = (
                (base_subblock_ids + index) if index > 0 else base_subblock_ids
            )
            # TODO: Potentially clamp all page 0 indices to the mask value.
            # Or even better, require that the ids are replicated such that access is
            # legal.
            # Now for each of the k/v attn_block_ids, which have been adjusted to
            # index into the sub-pages, we flatten to do a linear index_select
            # copy of the sub-blocks by collapsing the first two dims so we have
            # a linear list.
            subblock_table.index_copy_(
                0, subblock_ids.flatten(0, 1), part_block_view.flatten(0, 1)
            )

        for index, partition in enumerate(cache_partitions):
            write_cache_partition(index, partition)
