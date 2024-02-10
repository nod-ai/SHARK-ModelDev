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

from typing import Callable, Optional, Union

import math
import queue
from threading import Lock, Thread

from iree.runtime import (  # type: ignore[import-untyped]
    create_hal_module,
    get_driver,
    BufferUsage,
    HalBufferView,
    HalDevice,
    HalDriver,
    HalElementType,
    MemoryType,
    VmFunction,
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
        "_module_sets",
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
        self._module_sets: dict[str, "ModuleSet"] = {}
        self.queues = [WorkQueue(self, i) for i in range(queue_count)]

    def shutdown(self):
        for ms in self._module_sets.values():
            ms.shutdown()

    def create_module_set(self, name: str, *, context_count: int = 1) -> "ModuleSet":
        assert (
            name not in self._module_sets
        ), f"Modules with name {name} already created"
        lm = ModuleSet(self, name, context_count=context_count)
        self._module_sets[name] = lm
        return lm

    def module_set(self, name: str) -> "ModuleSet":
        try:
            return self._module_sets[name]
        except KeyError:
            raise KeyError(
                f"ModuleSet '{name}' not found. Available: {self._module_sets.keys()}"
            )


class ModuleSet:
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
            self.contexts[i] = HostContext(
                self.session, self.modules, name=f"HostContext-{self.name}-{i}"
            )

    def shutdown(self):
        for hc in self.contexts:
            hc.shutdown()

    def module(self, name: str) -> VmModule:
        for m in self.modules:
            if m.name == name:
                return m
        raise KeyError(
            f"Module {name} not found. Available: {[m.name for m in self.modules]}"
        )

    def function(self, module_name: str, function_name: str) -> VmFunction:
        m = self.module(module_name)
        f = m.lookup_function(function_name)
        if f is None:
            raise KeyError(
                f"Function '{function_name}' not found in '{module_name}'. "
                f"Available: {m.function_names}"
            )
        return f

    @property
    def context(self) -> "HostContext":
        """Gets a context, load balancing across available instances."""
        with _CONFIG_LOCK:
            self._context_counter += 1
            counter = self._context_counter
        contexts = self.contexts
        context = contexts[counter % len(contexts)]
        assert context is not None, "Module set not initialized"
        return context


class HostContext:
    def __init__(self, session: DeviceSession, modules: list[VmModule], name: str):
        self.session = session
        self.vm_context = VmContext(session.vm_instance, modules=modules)
        self.name = name
        self._thunk_queue = queue.SimpleQueue()
        t = Thread(
            target=_host_context_run,
            args=[self._thunk_queue, name],
            name=name,
            daemon=False,
        )
        self._running = True
        t.start()

    def shutdown(self):
        logger.info("Signalling shutdown of host context %s", self.name)
        self._thunk_queue.put(None)
        self._running = False

    def __del__(self):
        if self._running:
            self.shutdown()

    def schedule(self, thunk: Callable[[], None]):
        self._thunk_queue.put(thunk)


def _host_context_run(thunk_queue: queue.SimpleQueue, name: str):
    logger.info("Starting host context thread %s", name)
    try:
        while True:
            item = thunk_queue.get()
            if item is None:
                break
            try:
                item()
            except:
                # TODO: Notify some watchdog to shutdown.
                logger.exception(
                    "Unhandled exception on host context thread %s. "
                    "Processing will continue but the system needs "
                    "to reset.",
                    name,
                )
    except:
        logger.exception(
            "Host context thread %s died with a top-level exception!", name
        )
    finally:
        logger.info("Host context thread %s finished", name)


class WorkQueue:
    def __init__(self, session: DeviceSession, index: int):
        self.session = session
        self.index = index


class TransferBuffer:
    """Transfer buffers are pairs of host/device buffers of a specific size.

    They are used for streaming to/from the device.
    """

    __slots__ = [
        "host_buffer",
        "device_buffer",
    ]

    def __init__(self, session: DeviceSession, buffer_size_bytes: int):
        self.host_buffer = session.device.allocator.allocate_buffer(
            memory_type=MemoryType.HOST_LOCAL | MemoryType.DEVICE_VISIBLE,
            allowed_usage=BufferUsage.DEFAULT,
            allocation_size=buffer_size_bytes,
        )
        self.device_buffer = session.device.allocator.allocate_buffer(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=BufferUsage.DEFAULT,
            allocation_size=buffer_size_bytes,
        )

    @staticmethod
    def allocate_shaped(
        session: DeviceSession, shape: list[int], element_type: HalElementType
    ) -> "TransferBuffer":
        assert HalElementType.is_byte_aligned(element_type)
        buffer_size_bytes = math.prod(shape) * HalElementType.dense_byte_count(
            element_type
        )
        return TransferBuffer(session, buffer_size_bytes)


class TransferBufferPool:
    """Pool of transfer buffers of a fixed size."""

    __slots__ = [
        "_allocator",
        "_free_list",
        "name",
    ]

    def __init__(
        self,
        allocator: Callable[[], TransferBuffer],
        *,
        initial_capacity: int,
        growable: bool = False,
        name: str = "",
    ):
        self.name = name
        if initial_capacity > 0:
            self._free_list = [allocator() for _ in range(initial_capacity)]
        self._allocator = None
        if growable:
            self._allocator = allocator

    @staticmethod
    def shaped(
        session: DeviceSession,
        shape: list[int],
        element_type: HalElementType,
        *,
        initial_capacity: int,
        growable: bool = False,
        name: str = "",
    ) -> "TransferBufferPool":
        """Allocates a pool of transfer buffers of the given shape."""
        if initial_capacity > 0:
            logger.info(
                "Allocating initial capacity %s of '%s' transfer buffers: %s x %r",
                initial_capacity,
                name,
                shape,
                element_type,
            )
        return TransferBufferPool(
            lambda: TransferBuffer.allocate_shaped(session, shape, element_type),
            initial_capacity=initial_capacity,
            growable=growable,
            name=name,
        )

    @staticmethod
    def sized(
        session: DeviceSession,
        buffer_byte_size: int,
        *,
        initial_capacity: int,
        growable: bool = False,
        name: str = "",
    ) -> "TransferBufferPool":
        """Allocates a pool of transfer buffers of a given size in bytes."""
        if initial_capacity > 0:
            logger.info(
                "Allocating initial capacity %s of '%s' transfer buffers: %s bytes",
                initial_capacity,
                name,
                buffer_byte_size,
            )
        return TransferBufferPool(
            lambda: TransferBuffer(session, buffer_byte_size),
            initial_capacity=initial_capacity,
            growable=growable,
            name=name,
        )

    def acquire(self) -> TransferBuffer:
        """Acquires a transfer buffer from the pool.

        Must be returned via recycle() when done.
        """
        free_list = self._free_list
        if len(free_list) > 0:
            return free_list.pop()
        allocator = self._allocator
        if not allocator:
            raise RuntimeError(
                f"Transfer buffer pool '%s' exhausted and not growable", self.name
            )
        logger.info("Grow transfer buffer pool '%s'", self.name)
        return allocator()

    def recycle(self, tb: TransferBuffer):
        """Recycles an acquired transfer buffer."""
        self._free_list.append(tb)
