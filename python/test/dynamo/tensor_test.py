# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import time
import unittest

import numpy as np
import torch

# Public API imports.
from shark_turbine.dynamo import Device, TurbineMode, DeviceTensor


class TensorTest(unittest.TestCase):
    def setUp(self):
        self.mode = TurbineMode()
        self.mode.__enter__()
        Device("local-task").set()

    def tearDown(self) -> None:
        Device.current().clear()
        self.mode.__exit__(None, None, None)

    @unittest.expectedFailure
    def test_explicit_construct(self):
        size = (2, 2)
        t1 = DeviceTensor(size, torch.float32, np.ones(size))
        t2 = DeviceTensor(
            size, torch.float32, np.arange(size[0] * size[1]).reshape(size)
        )
        print("Inputs:")
        print(t1)
        print(t2)

    def test_async_copy_from_host(self):
        t1 = torch.empty(4, device="turbine")
        ar = np.arange(4, dtype=np.float32)
        t1._async_copy_from_host(ar)
        np.testing.assert_array_equal(t1.cpu(), ar)

    def test_cpu_to(self):
        t_cpu = torch.arange(4).cpu()
        t_t = t_cpu.to(device="turbine")
        np.testing.assert_array_equal(t_cpu, t_t.cpu())

    def test_factory_function_empty(self):
        # Factory functions
        t1 = torch.empty(4, device="turbine")
        print("Empty Tensor (un-initialized memory!):")
        print(t1)

    def test_factory_function_empty_tuple_size(self):
        # TODO: Test some invariants vs just printing.
        t1 = torch.empty((4, 4), device="turbine")
        print("Empty Tensor (un-initialized memory!):")
        print(t1)
        print(t1.buffer_view)
        print(t1.to("cpu"))
        print(t1.cpu())

    def test_factory_function_zeros(self):
        t1 = torch.zeros(2, 3, device="turbine")
        np.testing.assert_array_equal(t1.cpu(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def test_factory_function_ones(self):
        t1 = torch.ones(2, 3, device="turbine")
        np.testing.assert_array_equal(t1.cpu(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_factory_arange(self):
        t1 = torch.arange(4, device="turbine", dtype=torch.float32)
        ar = np.arange(4, dtype=np.float32)
        np.testing.assert_array_equal(t1.cpu(), ar)

    def test_factory_rand(self):
        t1 = torch.rand(4, device="turbine", dtype=torch.float32)
        print(t1.cpu())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
