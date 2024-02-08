# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime session constructs.

Key concepts:

  * DeviceSession: A single HAL device and other process-level globals. Shared global
    memory and corresponding synchronization handles are accessible from here.
  * WorkQueue: Logical stream of execution, nested under the DeviceSession. Each 
    queue holds a timeline semaphore which sequences invocations. For these models,
    we route workloads of vastly different characteristics to distinct queues (i.e.
    prefill vs decode step).
  * LoadedModule: Modules that have been loaded but have not yet been instantiated into
    a context.
  * HostContext: At least one HostContext is created per LoadedModule. It encapsulates
    a VMContext and performs invocations on a dedicated thread. Typically, there will
    be more that one HostContext per LoadedModule as it helps us load balance the
    host side work across multiple OS threads, ensuring faster feeding of the device.
"""

from typing import Optional, Union

from threading import Lock

from iree.runtime import (  # type: ignore[import-untyped]
    create_hal_module,
    get_driver,
    HalDevice,
    HalDriver,
    VmInstance,
    VmContext,
    VmModule,
)

from .logging import get_logger

logger = get_logger("shark_turbine.serving.session")
_CONFIG_LOCK = Lock()
_GLOBAL_VM_INSTANCE: Optional[VmInstance] = None


def get_vm_instance() -> VmInstance:
    global _GLOBAL_VM_INSTANCE
    if not _GLOBAL_VM_INSTANCE:
        with _CONFIG_LOCK:
            if not _GLOBAL_VM_INSTANCE:
                _GLOBAL_VM_INSTANCE = VmInstance()
    return _GLOBAL_VM_INSTANCE


class DeviceSession:
    """Top-level object associated with a single attached device."""

    __slots__ = [
        "device",
        "driver",
        "modules",
        "queues",
        "vm_instance",
    ]

    def __init__(
        self,
        *,
        uri: Optional[str] = None,
        driver: Optional[Union[str, HalDriver]] = None,
        device: Optional[HalDevice] = None,
        vm_instance: Optional[VmInstance] = None,
        queue_count: int = 1,
    ):
        self.vm_instance = vm_instance or get_vm_instance()
        if uri is not None:
            assert (
                driver is None and device is None
            ), "If 'uri' is given, 'driver' and 'device' cannot be set"
            logger.info("Opening device by uri: %s", uri)
            driver = uri_driver = get_driver(uri)
            device = uri_driver.create_device_by_uri(uri)
        assert driver is not None, "'driver' cannot be None"
        self.driver = driver if not isinstance(driver, str) else get_driver(driver)
        self.device = device if device else self.driver.create_default_device()

        # Dependent objects.
        self.modules: dict[str, "LoadedModules"] = {}
        self.queues = [WorkQueue(self, i) for i in range(queue_count)]

    def new_modules(self, name: str, *, context_count: int = 1) -> "LoadedModules":
        assert name not in self.modules, f"Modules with name {name} already created"
        lm = LoadedModules(self, name, context_count=context_count)
        self.modules[name] = lm
        return lm


class LoadedModules:
    __slots__ = [
        "contexts",
        "modules",
        "name",
        "session",
        "_context_counter",
    ]

    def __init__(self, session: DeviceSession, name: str, *, context_count: int):
        assert context_count > 0
        self.session = session
        self.name = name
        self.modules: list[VmModule] = [
            create_hal_module(session.vm_instance, session.device)
        ]
        self.contexts = [None] * context_count
        self._context_counter = 0

    @property
    def initialized(self) -> bool:
        return self.contexts[-1] is not None

    def add(self, module: VmModule):
        self.modules.append(module)

    def load_vmfb(self, vmfb_path: str):
        logger.info("Loading VMFB %s", vmfb_path)
        self.add(VmModule.mmap(self.session.vm_instance, vmfb_path))

    def initialize(self):
        assert not self.initialized, "Already initialized"
        count = len(self.contexts)
        logger.info("Initializing %s contexts for %s", count, self.name)
        for i in range(count):
            self.contexts[i] = HostContext(self.session, self)

    @property
    def context(self) -> "HostContext":
        """Gets a context, load balancing across available instances."""
        with _CONFIG_LOCK:
            self._context_counter += 1
            counter = self._context_counter
        contexts = self.contexts
        context = contexts[counter % len(contexts)]
        assert context is not None
        return context


class HostContext:
    def __init__(self, session: DeviceSession, modules: LoadedModules):
        self.session = session
        self.vm_context = VmContext(session.vm_instance, modules=modules.modules)


class WorkQueue:
    def __init__(self, session: DeviceSession, index: int):
        self.session = session
        self.index = index
