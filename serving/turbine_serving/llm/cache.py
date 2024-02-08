# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Manages the block cache."""

from .config import BlockCacheParams
from .session import DeviceSession


class BlockCache:
    def __init__(self, session: DeviceSession, params: BlockCacheParams):
        ...
