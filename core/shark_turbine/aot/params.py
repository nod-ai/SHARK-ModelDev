# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Iterator, Optional, Set, Tuple, Union

from pathlib import Path

import torch
import torch.nn as nn

from iree.runtime import (
    ParameterIndex,
)

################################################################################
# External tensor type
################################################################################


class ExternalTensor(torch.Tensor):
    """Tensor which is backed by a real tensor and carries metadata tieing
    it to an external parameter pack.

    See: https://github.com/albanD/subclass_zoo/blob/main/trivial_tensors.py
    """

    _is_turbine_external_tensor: bool
    external_scope: str
    external_name: str

    @staticmethod
    def __new__(
        cls, data, *, requires_grad=None, external_name: str, external_scope: str = ""
    ):
        if requires_grad is None:
            return torch.Tensor.__new__(cls, data)
        else:
            return cls._make_subclass(cls, data, requires_grad)

    def __init__(
        self, data, *, external_name: str, external_scope: str = "", requires_grad=None
    ):
        super().__init__()
        self.data = data
        if isinstance(data, nn.Parameter):
            # Magic that makes the instanceof check report as a Parameter.
            self._is_param = True
        self._is_turbine_external_tensor = True
        self.external_scope = external_scope
        self.external_name = external_name

    __torch_function__ = torch._C._disabled_torch_function_impl


def externalize_module_parameters(
    module: nn.Module, *, external_scope: str = "", prefix: str = ""
):
    """Externalizes parameters and persistent buffers in a module by name."""

    def externalize(state_dict: dict, prefix: str, item_name: str, item: torch.Tensor):
        full_name = f"{prefix}.{item_name}" if prefix else item_name
        external_item = ExternalTensor(
            item,
            requires_grad=item.requires_grad,
            external_scope=external_scope,
            external_name=full_name,
        )
        torch.utils.swap_tensors(item, external_item)

    memo: Set[str] = set()
    for sub_name, sub_module in module.named_modules(prefix=prefix):
        state_dict = sub_module.state_dict()
        for param_name, param in sub_module.named_parameters(recurse=False):
            if param_name in memo:
                continue
            if param_name not in state_dict:
                # Non persistent
                continue
            memo.add(param_name)
            externalize(state_dict, sub_name, param_name, param)
        for buffer_name, buffer in sub_module.named_buffers(recurse=False):
            if buffer_name in memo:
                continue
            memo.add(buffer_name)
            if buffer_name not in state_dict:
                # Non persistent
                continue
            externalize(state_dict, sub_name, buffer_name, buffer)


################################################################################
# Parameter archives save/load
################################################################################


def save_module_parameters(
    file_path: Union[str, Path], module: nn.Module, *, prefix: str = ""
):
    """One shot save of parameters and persistent buffers on a module.

    More options are available by using a ParameterArchiveBuilder.
    """
    builder = ParameterArchiveBuilder()
    builder.add_module(module, prefix=prefix)
    builder.save(file_path)


class ParameterArchive:
    """Allows access to a parameter archive as CPU tensors."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        self._index = ParameterIndex()
        if file_path is not None:
            self.load(file_path)

        # for i in range(len(self._index)):
        #     entry = self._index[i]
        #     file_handle, offset = entry.file_storage
        #     view = file_handle.view[offset:offset + entry.length]

    def load(
        self,
        file_path: Union[str, Path],
        *,
        readable: bool = True,
        writable: bool = False,
    ):
        """Loads index entries from a file adding them to the in-memory archive."""
        self._index.load(file_path, readable=readable, writable=writable)

    def __repr__(self):
        return repr(self._index)


class ParameterArchiveBuilder:
    """Helper for building parameter archives from live modules."""

    def __init__(self):
        self._index = ParameterIndex()

    def save(self, file_path: Union[str, Path]):
        """Saves the archive."""
        self._index.create_archive_file(str(file_path))

    def add_tensor(self, name: str, tensor: torch.Tensor):
        """Adds an named tensor to the archive."""
        host_array = tensor.detach().cpu().contiguous().numpy()
        self._index.add_buffer(name, host_array)

    def add_module(self, module: nn.Module, *, prefix: str = ""):
        """Adds all parameters and persistent buffers from a module hierarchy."""
        for name, t in self._yield_saveable_tensors(module, prefix=prefix):
            self.add_tensor(name, t)

    def _yield_saveable_tensors(
        self, module: nn.Module, *, prefix: str = ""
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        memo: Set[str] = set()
        for sub_name, sub_module in module.named_modules(prefix=prefix):
            state_dict = sub_module.state_dict()
            for param_name, param in sub_module.named_parameters(
                prefix=sub_name, recurse=False
            ):
                if param_name in memo:
                    continue
                memo.add(param_name)
                yield param_name, param
            for buffer_name, buffer in sub_module.named_buffers(
                prefix=sub_name, recurse=False
            ):
                if buffer_name in memo:
                    continue
                memo.add(buffer_name)
                if buffer_name not in state_dict:
                    # Non persistent
                    continue
                yield buffer_name, buffer
