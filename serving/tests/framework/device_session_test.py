# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from turbine_serving.framework.session import (
    DeviceSession,
)


@pytest.fixture
def local_device_session():
    session = DeviceSession(uri="local-task")
    yield session
    session.shutdown()


def test_start_shutdown_no_host_contexts(local_device_session: DeviceSession):
    ms = local_device_session.create_module_set("default")
    ms.initialize()


def test_host_context_start_stop(local_device_session: DeviceSession):
    ms = local_device_session.create_module_set("default")
    ms.initialize()
    hc = ms.host_context


def test_host_context_scheduling(local_device_session: DeviceSession):
    device = local_device_session.device
    ms = local_device_session.create_module_set("default")
    ms.initialize()
    hc = ms.host_context

    sem = device.create_semaphore(0)

    async def task1():
        print("[coro1] test_host_context_scheduling.task")
        await hc.on_semaphore(sem, 1, True)
        print("[coro1] await completed")
        sem.signal(2)

    async def task2():
        print("[coro2] waiting for 2")
        await hc.on_semaphore(sem, 2, True)
        sem.fail("Fail from task2")

    f1 = hc.run_concurrent(task1())
    f2 = hc.run_concurrent(task2())
    sem.signal(1)
    print("[main] Waiting for semaphore")

    # Ensure task completion. Important to consume to ensure that exceptions
    # propagate.
    f1.result()
    f2.result()

    print("[main] Waiting on semaphore payload 3")
    with pytest.raises(Exception, match="Fail from task2"):
        sem.wait(3)
