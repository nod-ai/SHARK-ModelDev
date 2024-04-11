# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, Union, Collection
from dataclasses import dataclass

from .inference_ops import InferenceOps
from .tensors import InferenceTensor

__all__ = [
    "Dataset",
    "Theta",
]

class Theta:
    """Subset of parameter tensors used for inference."""

    def __init__(
        self,
        tensors: dict,
        *,
        ops: Optional["InferenceOps"] = None,
        already_nested: bool = False,
    ):
        self._tensors = tensors if already_nested else _flat_to_nested_dict(tensors)
        self.ops = ops if ops is not None else InferenceOps()

    def flatten(self) -> dict[str, InferenceTensor]:
        results = {}

        def accum(prefix, child):
            for key, value in child.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    accum(new_prefix, value)
                else:
                    results[new_prefix] = value

        accum("", self._tensors)
        return results

    def tensor(self, *name_path: Union[str, int]) -> InferenceTensor:
        current_ts = self._tensors
        try:
            for part in name_path[0:-1]:
                current_ts = current_ts[str(part)]
            last = name_path[-1]
            t = current_ts[str(last)]
        except KeyError:
            raise KeyError(
                f"Unknown parameter {name_path} (in Theta object "
                f"containing {self.keys})"
            )
        return t

    @property
    def keys(self) -> Collection[str]:
        return self._tensors.keys()

    @property
    def tensors(self) -> Collection[InferenceTensor]:
        return self._tensors.values()

    def __call__(self, *name_path: Union[str, int]) -> "Theta":
        current_ts = self._tensors
        try:
            for part in name_path:
                current_ts = current_ts[str(part)]
        except KeyError:
            raise KeyError(f"Sub-theta {name_path} not found")
        return Theta(current_ts, ops=self.ops)

    def __repr__(self):
        return f"Theta({self.keys})"


def _flat_to_nested_dict(flat: dict[str, Any]) -> dict[str, Any]:
    nested: dict = {}

    def add_to_dict(
        name: str,
        value,
    ):
        current = nested

        parts = name.split(".")
        for part in parts[0:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            assert isinstance(
                current, dict
            ), f"Name collision in parameter dict: {name}"
        current[parts[-1]] = value

    for name, value in flat.items():
        add_to_dict(name, value)
    return nested


@dataclass
class Dataset:
    """Top level configuration for a model.

    This consists of:

    * Dataset level hyper-parameters (properties).
    * Root theta with materialized parameters (Theta).
    """

    properties: dict[str, Any]
    root_theta: Theta
