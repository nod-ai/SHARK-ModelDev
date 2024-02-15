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

from typing import Any, Callable, Coroutine, TypeVar, Optional, Union

import asyncio
import concurrent.futures
import math
import queue
from threading import Lock, Thread
import warnings

import numpy as np

from iree.runtime import (  # type: ignore[import-untyped]
    create_hal_module,
    get_driver,
    BufferUsage,
    HalBufferView,
    HalCommandBuffer,
    HalDevice,
    HalDeviceLoopBridge,
    HalDriver,
    HalElementType,
    HalFence,
    HalSemaphore,
    MemoryType,
    VmFunction,
    VmInstance,
    VmContext,
    VmModule,
)

from .logging import get_logger, NDEBUG

T = TypeVar("T")

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
        "_queue_request_count",
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
        self._queue_request_count = 0
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

    def queue(self, index: int = -1) -> "WorkQueue":
        """Gets a queue either with an explicit index or in some rotating fashion."""
        if index >= 0:
            return self.queues[index]
        else:
            self._queue_request_count += 1
            qc = self._queue_request_count
            return self.queues[qc % len(self.queues)]


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
            if hc is not None:
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
    def host_context(self) -> "HostContext":
        """Gets a context, load balancing across available instances."""
        with _CONFIG_LOCK:
            self._context_counter += 1
            counter = self._context_counter
        contexts = self.contexts
        context = contexts[counter % len(contexts)]
        assert context is not None, "Module set not initialized"
        return context


_ThunkQueueT = queue.SimpleQueue[Union[None, Callable[[], None]]]


class HostContext:
    def __init__(self, session: DeviceSession, modules: list[VmModule], name: str):
        self.session = session
        self.vm_context = VmContext(session.vm_instance, modules=modules)
        self.name = name
        self.loop = asyncio.new_event_loop()
        self.loop.set_debug(True)

        # def exc_handler(loop, context):
        #     print("[EXCEPTION]", loop, context)
        # self.loop.set_exception_handler(exc_handler)

        self._device_bridge = HalDeviceLoopBridge(session.device, self.loop)
        self._shutdown_future = self.loop.create_future()
        logger.info(f"Starting asyncio loop thread %s", name)
        self._loop_thread = Thread(
            target=self.loop.run_until_complete,
            args=[self._shutdown_future],
            name=name,
            daemon=False,
        )
        self._loop_thread.start()

    def shutdown(self, join: bool = True):
        if self._shutdown_future is None:
            return
        logger.info("Signalling shutdown of host context %s", self.name)
        local_future = self._shutdown_future
        del self._shutdown_future

        def _shutdown():
            local_future.set_result(True)

        self.loop.call_soon_threadsafe(_shutdown)
        self._device_bridge.stop()
        if join:
            self._loop_thread.join()
        self.loop.close()

    def __del__(self):
        if hasattr(self, "_shutdown_future"):
            warnings.warn(f"HostContext deallocated without shutdown(): {self}")
            self.shutdown(join=False)

    def run_concurrent(
        self, coro: Coroutine[Any, Any, T]
    ) -> concurrent.futures.Future[T]:
        """Runs a coroutine from another thread, returning a concurrent Future.

        This should be used for submitting initial work to the host context from
        another thread or event loop.

        Note that the concurrent Future should have its result() retrieved to
        ensure that any asynchronous exceptions are propagated. Otherwise, they will
        be silently consumed.
        """
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """Runs a coroutine on the host context loop from another thread.

        Waits on and returns the result.
        This is primarily intended for testing.
        """
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def on_semaphore(
        self, sem: HalSemaphore, payload: int, value: Any
    ) -> asyncio.Future:
        """Returns an awaitable for when the semaphore attains a payload timepoint.

        The resulting Future will take the given `value` once complete.
        """
        return self._device_bridge.on_semaphore(sem, payload, value)


class WorkQueue:
    """Models a queue as a progression of steps against a timeline semaphore."""

    __slots__ = [
        "_device",
        "_lock",
        "_semaphore",
        "_step",
        "index",
    ]

    def __init__(self, session: DeviceSession, index: int = 0):
        self.index = index
        self._device = session.device
        self._lock = Lock()
        self._semaphore = session.device.create_semaphore(0)
        self._step = 0

    def execute_sequential(self, command_buffers: list[HalCommandBuffer]):
        """Executes a list of command buffers at the current step, advancing to the
        next.
        """
        with self._lock:
            current_step = self._step
            next_step = current_step + 1
            self._step = next_step
        sem = self._semaphore
        self._device.queue_execute(
            command_buffers, [(sem, current_step)], [(sem, next_step)]
        )

    def current_fence(self) -> HalFence:
        """Gets a fence representing the current step."""
        with self._lock:
            return HalFence.create_at(self._semaphore, self._step)

    def step_fences(self) -> tuple[HalFence, HalFence]:
        """Gets two fences, one at the current step and one at the next."""
        with self._lock:
            current_step = self._step
            next_step = current_step + 1
            self._step = next_step
        sem = self._semaphore
        return HalFence.create_at(sem, current_step), HalFence.create_at(sem, next_step)

    def sync(self, host_context: HostContext) -> asyncio.Future:
        """Awaitable that completes when all work currently queued completed."""
        with self._lock:
            current_step = self._step
        return host_context.on_semaphore(self._semaphore, current_step, True)

    def __repr__(self):
        with self._lock:
            return f"WorkQueue[{self.index}](semaphore={self._semaphore}, step={self._step}"


class TransferBuffer:
    """Transfer buffers are pairs of host/device buffers of a specific size.

    They are used for streaming to/from the device.
    """

    __slots__ = [
        "host_buffer",
        "device_buffer",
        "host_buffer_map",
        "_pool",
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
        self.host_buffer_map = self.host_buffer.map()
        self._pool: Optional["TransferBufferPool"] = None

    @staticmethod
    def allocate_shaped(
        session: DeviceSession, shape: list[int], element_type: HalElementType
    ) -> "TransferBuffer":
        assert HalElementType.is_byte_aligned(element_type)
        buffer_size_bytes = math.prod(shape) * HalElementType.dense_byte_count(
            element_type
        )
        return TransferBuffer(session, buffer_size_bytes)

    def recycle(self):
        pool = self._pool
        assert (
            pool is not None
        ), f"Cannot recycle a TransferBuffer that was not acquired from a pool ({self})"
        self._pool = None
        pool.recycle(self)

    def h2d_array(
        self,
        cb: HalCommandBuffer,
        shape: list[int],
        element_type: HalElementType,
        *,
        fill_value: Any = None,
    ) -> tuple[np.ndarray, HalBufferView]:
        """Performs an h2d transfer on the given CommandBuffer of the given shape and
        element type.

        Returns a host array and device buffer view. Because transfers do not start
        until the command buffer is submitted, the host array should be populated
        between the return from this call and submission.
        """
        ary = self.host_buffer_map.asarray(
            shape, HalElementType.map_to_dtype(element_type)
        )
        if fill_value is not None:
            ary.fill(fill_value)
        bv = HalBufferView(self.device_buffer, shape, element_type)
        cb.copy(self.host_buffer, self.device_buffer, length=bv.byte_length)
        return ary, bv

    def __repr__(self):
        if self._pool is None:
            return f"TransferBuffer(FREE)"
        else:
            return f"TransferBuffer({self._pool})"

    if not NDEBUG:

        def __del__(self):
            if self._pool is not None:
                warnings.warn(
                    f"Deallocated TransferBuffer which needed to be recycled: {self}"
                )


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
            tb = free_list.pop()
            assert tb._pool is None
            tb._pool = self
            return tb

        allocator = self._allocator
        if not allocator:
            raise RuntimeError(
                f"Transfer buffer pool '%s' exhausted and not growable", self.name
            )
        logger.info("Grow transfer buffer pool '%s'", self.name)
        tb = allocator()
        assert tb._pool is None
        tb._pool = self
        return tb

    def recycle(self, tb: TransferBuffer):
        """Recycles an acquired transfer buffer."""
        self._free_list.append(tb)

    def __repr__(self):
        return f"TransferBufferPool({self.name})"


class AsyncResources:
    """Resources held for some asynchronous scope."""

    __slots__ = [
        "_resources",
    ]

    def __init__(self):
        self._resources: list[Union[TransferBuffer, "AsyncResources"]] = []

    def acquire_transfer_buffer(self, pool: TransferBufferPool) -> TransferBuffer:
        tb = pool.acquire()
        self._resources.append(tb)
        return tb

    def recycle(self):
        for r in self._resources:
            r.recycle()
        self._resources.clear()

    if not NDEBUG:

        def __del__(self):
            if len(self._resources) != 0:
                warnings.warn(
                    f"Deallocated AsyncResources that was not recycled: {self}"
                )
                self.recycle()
