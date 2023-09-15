# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

from torch.utils._pytree import (
    TreeSpec,
    tree_flatten,
    tree_map,
    tree_unflatten,
)

import torch.nn as nn

from ..support import (
    AbstractTypedef,
    Abstractifiable,
    GlobalsDef,
    TreeAbstractifiable,
    abstractify_single_value,
)


class export_global(GlobalsDef, Abstractifiable):
    """Exports a single global into a CompiledModule."""

    __slots__ = ["_name", "_value"]

    def __init__(
        self,
        value: Any,
        *,
        name: str = "global",
        initialize: bool = True,
        mutable: bool = False,
    ):
        super().__init__(initialize=initialize, mutable=mutable)
        self._name = name
        self._value = value

    def items(self):
        yield (self._name, self._value)

    def abstractify(self) -> AbstractTypedef:
        return abstractify_single_value(self._value)


class export_parameters(GlobalsDef, TreeAbstractifiable):
    """Exports parameters from an nn.Module.

    These are exposed to procedural programs as a dictionary of param/values.
    """

    __slots__ = [
        "_param_list",
        "_schema",
        "_tree",
    ]

    def __init__(
        self, nn_module: nn.Module, *, initialize: bool = True, mutable: bool = False
    ):
        super().__init__(initialize=initialize, mutable=mutable)
        self._param_list = list(nn_module.named_parameters())
        self._tree = dict(self._param_list)
        _, self._schema = tree_flatten(self._tree)

    def items(self):
        for name, value in self._param_list:
            yield (name, value)

    def schema(self) -> TreeSpec:
        return self._schema

    def abstractify_tree(self):
        return tree_map(abstractify_single_value, self._tree)

    def __getitem__(self, key):
        return self._tree[key]

    def __repr__(self):
        names = [name for name, _ in self._param_list]
        return f"<export_parameters {', '.join(names)}>"
