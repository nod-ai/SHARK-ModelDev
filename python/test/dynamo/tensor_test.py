# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import numpy as np
import torch

# Public API imports.
from shark_turbine.dynamo import Device, TurbineMode, TurbineTensor


class TensorTest(unittest.TestCase):
    def setUp(self):
        Device("local-task").set()

    def tearDown(self) -> None:
        Device.current().clear()

    @unittest.expectedFailure
    def test_explicit_construct(self):
        size = (2, 2)
        t1 = TurbineTensor(size, torch.float32, np.ones(size))
        t2 = TurbineTensor(
            size, torch.float32, np.arange(size[0] * size[1]).reshape(size)
        )
        print("Inputs:")
        print(t1)
        print(t2)

    def test_factory_function_empty(self):
        # Factory functions
        t1 = torch.empty(4, device="turbine")
        print("Empty Tensor (un-initialized memory!):")
        print(t1)

    def test_factory_function_empty_tuple_size(self):
        # Factory functions
        t1 = torch.empty((4, 4), device="turbine")
        print("Empty Tensor (un-initialized memory!):")
        print(t1)
        print(t1.buffer_view)
        print(t1.to("cpu"))
        print(t1.cpu())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with TurbineMode():
        unittest.main()
