# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

import torch.nn as nn

from ..builder import GlobalsDef


class export_global(GlobalsDef):
    """Exports a single global into a CompiledModule."""

    __slots__ = ["_name", "_value"]

    def __init__(self, value: Any, *, name: str = "global"):
        self._name = name
        self._value = value

    def items(self):
        yield (self._name, self._value)


class export_parameters(GlobalsDef):
    """Exports parameters from an nn.Module."""

    def __init__(self, nn_module: nn.Module):
        self._param_list = list(nn_module.named_parameters())

    def items(self):
        for name, value in self._param_list:
            yield (name, value)

    def __repr__(self):
        names = [name for name, _ in self._param_list]
        return f"<export_parameters {', '.join(names)}>"
