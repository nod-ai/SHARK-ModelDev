# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Turbine tensor.

This implementation is adapted from a variety of sources, most notably the subclass
zoo: https://github.com/albanD/subclass_zoo/blob/main/new_device.py
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
from torch.overrides import TorchFunctionMode

from .device import (
    Device,
)

from ..support import (
    ApiSequencingError,
    UnknownDTypeError,
)

from iree.runtime import (
    HalBuffer,
    HalBufferView,
    HalElementType,
    HalFence,
    HalSemaphore,
)


_DTYPE_TO_ELEMENT_TYPE: Dict[torch.dtype, HalElementType] = {
    torch.float16: HalElementType.FLOAT_16,
    torch.bfloat16: HalElementType.BFLOAT_16,
    torch.float32: HalElementType.FLOAT_32,
    torch.float64: HalElementType.FLOAT_64,
    torch.uint8: HalElementType.UINT_8,
    torch.int8: HalElementType.SINT_8,
    torch.int16: HalElementType.SINT_16,
    torch.int32: HalElementType.SINT_32,
    torch.int64: HalElementType.SINT_64,
    torch.bool: HalElementType.BOOL_8,
    torch.qint8: HalElementType.OPAQUE_8,
    torch.quint8: HalElementType.OPAQUE_8,
    torch.complex64: HalElementType.COMPLEX_64,
    torch.complex128: HalElementType.COMPLEX_128,
}


def _dtype_to_element_type(dtype) -> HalElementType:
    try:
        return _DTYPE_TO_ELEMENT_TYPE[dtype]
    except KeyError:
        raise UnknownDTypeError(dtype)


_TORCH_DTYPE_TO_NUMPY = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def _torch_dtype_to_numpy(torch_dtype: torch.dtype) -> Any:
    try:
        return _TORCH_DTYPE_TO_NUMPY[torch_dtype]
    except KeyError:
        raise UnknownDTypeError(torch_dtype)


_ELEMENT_TYPE_TO_NUMPY_DTYPE = {
    HalElementType.FLOAT_16: np.float16,
    HalElementType.FLOAT_32: np.float32,
    HalElementType.FLOAT_64: np.float64,
    HalElementType.UINT_8: np.uint8,
    HalElementType.SINT_8: np.int8,
    HalElementType.SINT_16: np.int16,
    HalElementType.SINT_32: np.int32,
    HalElementType.SINT_64: np.int64,
    HalElementType.BOOL_8: np.bool_,
    HalElementType.COMPLEX_64: np.complex64,
    HalElementType.COMPLEX_128: np.complex128,
}


def _element_type_to_numpy_dtype(element_type: HalElementType) -> Any:
    try:
        return _DTYPE_TO_ELEMENT_TYPE[element_type]
    except KeyError:
        raise UnknownDTypeError(element_type)


###############################################################################
# Turbine storage
###############################################################################


class Storage:
    __slots__ = [
        "buffer",
        "device",
        "ready_fence",
        "done_fence",
    ]

    def __init__(self, device: Device, buffer: HalBuffer):
        fence_capacity = device._fence_capacity
        self.buffer = buffer
        self.device = device
        # Signalled when the buffer is ready to be consumed. Consumers should
        # join this fence and wait on it.
        self.ready_fence = HalFence(fence_capacity)
        # Signalled when all scheduled work on the buffer is done. Consumers
        # should advance this fence when mutations have been queued.
        self.done_fence = HalFence(fence_capacity)

    def sync(self):
        """Stops the world and waits for all scheduled mutations to complete."""
        self.done_fence.wait()

    def kill(self):
        """Kills the device memory associated with this storage."""
        if not self.buffer:
            raise ApiSequencingError("Storage.kill() called on a non-live instance")
        device = self.device
        hal_device = device._s.device
        hal_device.queue_dealloca(self.buffer, self.done_fence, [])
        self.buffer = None
        self.device = None

    def __del__(self):
        if self.buffer:
            self.kill()


###############################################################################
# Tensor class and support
###############################################################################


class TurbineTensor(torch.Tensor):
    """A Tensor accessing memory on a Turbine device."""

    @staticmethod
    def __new__(cls, size, dtype, raw_data=None, requires_grad=False):
        # Using a meta tensor as the wrapped gives us shape and dtype
        # propagation.
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(size, dtype=dtype, device="meta"),
            require_grad=requires_grad,
        )

    def __init__(self, size, dtype, raw_data=None, requires_grad=False):
        if isinstance(raw_data, Storage):
            self._storage = raw_data
            self._bv = None
        else:
            if raw_data is not None:
                raise NotImplementedError(
                    f"raw_data= not implemented for TurbineTensor ({raw_data.__class__})"
                )

    @property
    def buffer_view(self) -> HalBufferView:
        if self._bv is None:
            self._bv = HalBufferView(
                self._storage.buffer,
                shape=self.size(),
                element_type=_dtype_to_element_type(self.dtype),
            )
        return self._bv

    def cpu(self):
        return self.to("cpu")

    def __repr__(self):
        try:
            return f"<TurbineTensor(Device) of {self.buffer_view}>"
        except UnknownDTypeError:
            return f"<TurbineTensor(Device) of invalid dtype at {self._storage.buffer}>"


def _calculate_c_contig_size(size, dtype) -> int:
    """Calculates a C-contiguous buffer size in bytes for torch size and dtype."""
    assert isinstance(dtype, torch.dtype)
    # TODO: This is a user-level size so it can technically be a sequence.
    # Is this true?
    if len(size) == 1 and not isinstance(size[0], int):
        return _calculate_c_contig_size(size[0], dtype)
    accum = 0
    dtype_size = dtype.itemsize
    for s in size:
        accum += s * dtype_size
    return accum


###############################################################################
# Factories and device enablement
###############################################################################


class TurbineMode(TorchFunctionMode):
    """Enables PyTorch tensor device= support for Tensor factory functions.

    This can be used in a `with` block to dynamically scope enablement, or
    it can be enabled globally via the `enable()` function.
    """

    IMPLEMENTATIONS = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        def super_fn(*args, **kwargs):
            # Disable torch_function by hand because we don't want the wrapping behavior of
            # the super() impl
            with torch._C.DisableTorchFunction():
                return func(*args, **kwargs)

        if func in self.IMPLEMENTATIONS:
            try:
                return self.IMPLEMENTATIONS[func](super_fn, *args, **kwargs or {})
            except Exception as e:
                raise e

        # This is just a no-op for all the non-factory functions:
        return super_fn(*args, **kwargs or {})


def enable():
    """Enables PyTorch tensor device= support for Turbine permanently."""
    TurbineMode().__enter__()


# Convenient wrapper to register functions
def implements_factory(func):
    def _inner_fn(impl):
        TurbineMode.IMPLEMENTATIONS[func] = impl
        return impl

    return _inner_fn


_TURBINE_PREFIX = "turbine-"


def _parse_device(device_arg) -> Optional[Device]:
    if isinstance(device_arg, Device):
        return device_arg
    elif isinstance(device_arg, str):
        if device_arg == "turbine":
            return Device.current()
        elif device_arg.startswith(_TURBINE_PREFIX):
            return Device(device_arg[len(_TURBINE_PREFIX) :])


# And some factory functions
# By hand
@implements_factory(torch.Tensor.to)
def to(super_fn, self, device):
    # Note that we only implement a subset of .to() here
    turbine_device = _parse_device(device)
    if turbine_device:
        # To turbine.
        return TurbineTensor(self.size(), self.dtype, self.numpy())
    elif isinstance(self, TurbineTensor):
        # From turbine.
        # TODO: We can handle certain catwalk cases from/to specific device classes
        # before just falling back to transferring through the CPU.
        # Stop the world and transfer to CPU.
        storage = self._storage
        storage.sync()
        bv = self.buffer_view
        dtype_descr = HalElementType.map_to_dtype(bv.element_type)
        memory = storage.buffer.map()
        np_array = memory.asarray(self.size(), dtype_descr)
        return torch.from_numpy(np_array)
    else:
        return super_fn(self, device)


@implements_factory(torch.empty)
def empty(super_fn, *args, **kwargs):
    device: Optional[Device] = None
    device_spec = kwargs.get("device", None)
    if device_spec:
        device = _parse_device(device_spec)
    if not device:
        return super_fn(*args, **kwargs)

    # Turbine empty.
    size = args
    # TODO: Technically can be overriden by torch.set_default_tensor_type.
    # Not sure if we care.
    dtype = kwargs.get("dtype", torch.float32)
    alloc_size = _calculate_c_contig_size(size, dtype)

    hal_device = device._s.device
    # Async allocate a buffer, waiting for the device (tx_timeline, tx_timepoint)
    # and signalling tx_timepoint + 1. Because we are just creating an empty
    # (uninitialized) tensor, it is ready when allocation completes.
    tx_semaphore = device._tx_timeline
    current_tx_timepoint = device._tx_timepoint
    wait_semaphores = [(tx_semaphore, current_tx_timepoint)]
    alloca_complete_semaphore = (tx_semaphore, current_tx_timepoint + 1)
    signal_semaphores = [alloca_complete_semaphore]
    device._tx_timepoint += 1
    buffer = hal_device.queue_alloca(alloc_size, wait_semaphores, signal_semaphores)
    storage = Storage(device, buffer)
    storage.ready_fence.insert(
        alloca_complete_semaphore[0], alloca_complete_semaphore[1]
    )
    return TurbineTensor(*size, dtype=dtype, raw_data=storage)


# Have a nicer way to add many factories
def get_factory_wrapper(func):
    def inner(super_fn, size, **kwargs):
        turbine_device = None
        device = kwargs.get("device", None)
        if device:
            turbine_device = _parse_device(device)

        if not turbine_device:
            return super_fn(size, **kwargs)

        return TurbineTensor(size, kwargs.get("dtype", torch.float32), func(size))

    return inner


# implements_factory(torch.rand)(get_factory_wrapper(np.random.rand))
# implements_factory(torch.arange)(get_factory_wrapper(np.arange))
