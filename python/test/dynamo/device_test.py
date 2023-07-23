# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
import threading

# Public API imports.
from shark_turbine.dynamo import (
    Device,
)

# Internals.
from shark_turbine.dynamo.device import (
    _CURRENT_THREAD,
)

from shark_turbine.support.exceptions import *


class DeviceTest(unittest.TestCase):
    def test_create(self):
        d = Device("local-task")
        self.assertEqual(repr(d), "<Turbine Device: local-task>")

    def test_current_device(self):
        with self.assertRaises(NoCurrentDeviceError):
            Device.current()

        d1 = Device("local-task")
        d2 = Device("local-sync")
        with d1:
            self.assertIs(Device.current(), d1)

            with d2:
                self.assertIs(Device.current(), d2)

            self.assertIs(Device.current(), d1)

        with self.assertRaises(NoCurrentDeviceError):
            Device.current()

    def test_set_clear(self):
        d1 = Device("local-task")
        d2 = Device("local-sync")

        with self.assertRaises(MismatchedDeviceSetClearError):
            d1.clear()
        try:
            d1.set()
            self.assertIs(Device.current(), d1)
            with self.assertRaises(MismatchedDeviceSetClearError):
                d2.clear()
            d1.clear()
            with self.assertRaises(NoCurrentDeviceError):
                Device.current()
        finally:
            # Patch it back to the reset state for testing.
            _CURRENT_THREAD.stack = []

    def test_cached_devices_same_thread(self):
        d1 = Device("local-task")
        d2 = Device("local-task")
        self.assertIs(d1, d2)

    def test_cached_device_diff_thread(self):
        devices = [None, None]

        def run_t1():
            devices[0] = Device("local-task")

        def run_t2():
            devices[1] = Device("local-task")

        t1 = threading.Thread(target=run_t1)
        t2 = threading.Thread(target=run_t2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertIsNotNone(devices[0])
        self.assertIsNotNone(devices[1])
        self.assertIsNot(devices[0], devices[1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
