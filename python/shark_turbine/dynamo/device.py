# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import lru_cache
from typing import List, Optional, Sequence, Union
from threading import local, Lock

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

from ..support.exceptions import *

__all__ = [
    "get_vm_instance",
    "DeviceState",
]

_CONFIG_LOCK = Lock()
_GLOBAL_VM_INSTANCE: Optional[VmInstance] = None


def get_vm_instance() -> VmInstance:
    global _GLOBAL_VM_INSTANCE
    if not _GLOBAL_VM_INSTANCE:
        with _CONFIG_LOCK:
            if not _GLOBAL_VM_INSTANCE:
                _GLOBAL_VM_INSTANCE = VmInstance()
    return _GLOBAL_VM_INSTANCE


class DeviceState:
    """State for an instantiated HAL device.

    Note that the IREE runtime internally manages a global cache of drivers for
    standard named-access (not custom-constructed) drivers.
    """

    __slots__ = [
        "device",
        "driver",
        "instance",
    ]

    def __init__(
        self,
        *,
        driver: Union[str, HalDriver],
        device: Optional[HalDevice] = None,
        vm_instance: Optional[VmInstance] = None,
    ):
        self.instance = vm_instance or get_vm_instance()
        self.driver = driver if isinstance(driver, HalDriver) else get_driver(driver)
        self.device = device if device else self.driver.create_default_device()

    @staticmethod
    @lru_cache(maxsize=None)
    def from_uri(uri: str) -> "DeviceState":
        driver = get_driver(uri)
        return DeviceState(driver=driver, device=driver.create_device_by_uri(uri))


_CURRENT_THREAD = local()


class Device:
    """Represents a low-level device (HalDriver/HalDevice) and scheduling data.

    This is the type that user's interact with as a 'Device'. Devices can be handled
    loose-leaf or bound to a thread with a context manager.
    """

    __slots__ = [
        "_s",
        "_main_timeline",
        "_main_timepoint",
        "_tx_timeline",
        "_tx_timepoint",
        "_fence_capacity",
    ]

    def __new__(
        cls, uri: Optional[str] = None, *, device_state: Optional[DeviceState] = None
    ):
        if uri is not None:
            # Construction by URI is cached on the thread.
            assert not device_state, "device_state= cannot be given with explicit URI"
            try:
                existing = _CURRENT_THREAD.device_by_uri[uri]
            except (AttributeError, KeyError):
                ...
            else:
                return existing

            # New instance.
            device_state = DeviceState.from_uri(uri)
            new_inst = super().__new__(cls)
            new_inst._s = device_state
            try:
                _CURRENT_THREAD.device_by_uri[uri] = new_inst
            except AttributeError:
                _CURRENT_THREAD.device_by_uri = {uri: new_inst}
            new_inst._initialize()
            return new_inst
        else:
            # Explicit construction with a device_state is assumed that you know what you
            # are doing and an uncached instance will be returned. This will be unsychronized
            # relative to any cached instance.
            assert device_state, "device_state= must be given if URI ommitted"
            new_inst = super().__new__(cls)
            new_inst._s = device_state
            new_inst._initialize()
            return new_inst

    def _initialize(self):
        d = self._s.device
        self._main_timeline = d.create_semaphore(0)
        self._main_timepoint = 0
        self._tx_timeline = d.create_semaphore(0)
        self._tx_timepoint = 0
        # Maximum number of semaphores the device uses. Can be increased if doing out of the
        # ordinary scheduling.
        self._fence_capacity = 2

    @property
    def hal_device(self) -> HalDevice:
        return self._s.device

    def current() -> "Device":
        try:
            return _CURRENT_THREAD.stack[-1]
        except (AttributeError, IndexError):
            raise NoCurrentDeviceError()

    def set(self) -> "Device":
        """Sets this device as the current device without a context manager."""
        try:
            _CURRENT_THREAD.stack.append(self)
        except AttributeError:
            _CURRENT_THREAD.stack = [self]

    def clear(self):
        """Clears the current device without a context manager."""
        try:
            c = _CURRENT_THREAD.stack[-1]
            if _CURRENT_THREAD.stack[-1] is self:
                _CURRENT_THREAD.stack.pop()
                return
        except (AttributeError, IndexError):
            ...
        raise MismatchedDeviceSetClearError()

    def __repr__(self):
        return f"<Turbine Device: {self._s.device}>"

    def __enter__(self):
        try:
            _CURRENT_THREAD.stack.append(self)
        except AttributeError:
            _CURRENT_THREAD.stack = [self]

    def __exit__(self, type, value, traceback):
        _CURRENT_THREAD.stack.pop()
