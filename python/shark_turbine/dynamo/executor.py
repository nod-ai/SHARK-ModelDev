# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from typing import List, Optional, Sequence, Union

from iree.runtime import (
    asdevicearray,
    create_hal_module,
    HalBufferView,
    DeviceArray,
    get_driver,
    VmContext,
    HalDevice,
    HalDriver,
    VmInstance,
    VmModule,
    VmVariantList,
)

from torch import (
    from_numpy as torch_from_numpy,
)

from .device import DeviceState


@functools.lru_cache(maxsize=None)
def get_vm_instance() -> VmInstance:
    return VmInstance()


class SpecializedExecutable:
    """A concrete executable that has been specialized in some way."""

    __slots__ = [
        "device_state",
        "entry_function",
        "user_module",
        "vm_context",
    ]

    def __init__(
        self,
        user_module: VmModule,
        device_state: DeviceState,
        entry_name: str = "main",
    ):
        self.user_module = user_module
        self.vm_context = VmContext(
            device_state.instance,
            (
                create_hal_module(device_state.instance, device_state.device),
                user_module,
            ),
        )
        self.device_state = device_state
        self.entry_function = self.user_module.lookup_function(entry_name)

    def __call__(self, *inputs):
        arg_list = VmVariantList(len(inputs))
        ret_list = VmVariantList(
            1
        )  # TODO: Get the number of results from the descriptor.

        # Move inputs to the device and add to arguments.
        self._inputs_to_device(inputs, arg_list)
        # TODO: Append semaphores for async execution.

        # Invoke.
        self.vm_context.invoke(self.entry_function, arg_list, ret_list)
        return self._returns_to_user(ret_list)

    def _inputs_to_device(self, inputs: list, arg_list: VmVariantList):
        # TODO: We are assuming the worst case here which is that we have unknown Torch
        # tensors that we send to the CPU and make continguous. Ideally, we would have
        # fast paths for our own backends and interop.
        for input in inputs:
            input_cpu = input.cpu().contiguous()
            # Since this is already a fallback case, just use the numpy array interop.
            # It isn't great, but meh... fallback case.
            device_array = asdevicearray(self.device_state.device, input_cpu)
            arg_list.push_ref(device_array._buffer_view)

    def _returns_to_user(self, ret_list: VmVariantList):
        # TODO: This is also not good that we are moving back to the CPU like this.
        # We should be returning a custom Tensor implementation which represents
        # our device data and has synchronization hooks for accessing it.
        device = self.device_state.device
        num_returns = len(ret_list)
        user_returns = [None] * num_returns
        for i in range(num_returns):
            device_buffer_view = HalBufferView.__iree_vm_cast__(ret_list.get_as_ref(i))
            device_array = DeviceArray(device, device_buffer_view)
            host_array = device_array.to_host()
            user_returns[i] = torch_from_numpy(host_array)

        return user_returns
