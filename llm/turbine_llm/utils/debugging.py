# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tools for debugging models."""

from dataclasses import dataclass
import re
import os
from typing import Sequence

import torch

from .logging import get_logger

__all__ = []

logger = get_logger("turbine_llm.debugging")

FLAGS_ENV_NAME = "TURBINE_LLM_DEBUG"
SETTING_PART_PATTERN = re.compile(r"""^([\\+\\-])?([^=]+)(=(.*))?$""")


@dataclass
class DebugFlags:
    enable_tensor_trace: bool = False

    def set(self, part: str):
        m = re.match(SETTING_PART_PATTERN, part)
        if not m:
            logger.warn("Syntax error in %s flag: '%s'", FLAGS_ENV_NAME, part)
            return
        logical_sense = m.group(1) != "-"
        name = m.group(2)
        value = m.group(4)

        if name == "tensor_trace":
            self.enable_tensor_trace = logical_sense
        else:
            logger.warn("Unrecognized %s flag: '%s'", FLAGS_ENV_NAME, name)

    @staticmethod
    def parse(settings: str) -> "DebugFlags":
        new_flags = DebugFlags()
        parts = settings.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            new_flags.set(part)
        return new_flags

    @staticmethod
    def parse_from_env() -> "DebugFlags":
        settings = os.getenv(FLAGS_ENV_NAME)
        if settings is None:
            return DebugFlags()
        new_flags = DebugFlags.parse(settings)
        logger.debug("Parsed debug flags from env %s: %r", FLAGS_ENV_NAME, new_flags)
        return new_flags


flags = DebugFlags.parse_from_env()


def trace_tensor(key: str, t: torch.Tensor):
    if not flags.enable_tensor_trace:
        return
    print(f"::: TRACE {key}({list(t.shape), t.dtype}) = {t}")
