# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import unittest
import torch


def add(x):
    return x + 8.2


def sub(x):
    return x - 1.6


def mul(x):
    return x * 3.2


def div(x):
    return x / 2.1


def floor_div(x):
    return x // 4.2


class TensorScalrOpConversionSmokeModule(unittest.TestCase):
    def setUp(self):
        self.t = torch.randn(2, 2)

    def test_add(self):
        opt_torch_scalar_convert = torch.compile(add, backend="turbine_cpu")
        result = opt_torch_scalar_convert(self.t)
        expected_result = add(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def test_sub(self):
        opt_torch_scalar_convert = torch.compile(sub, backend="turbine_cpu")
        result = opt_torch_scalar_convert(self.t)
        expected_result = sub(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def test_mul(self):
        opt_torch_scalar_convert = torch.compile(mul, backend="turbine_cpu")
        result = opt_torch_scalar_convert(self.t)
        expected_result = mul(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def test_div(self):
        opt_torch_scalar_convert = torch.compile(div, backend="turbine_cpu")
        result = opt_torch_scalar_convert(self.t)
        expected_result = div(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    @unittest.expectedFailure
    def test_floor_div(self):
        opt_torch_scalar_convert = torch.compile(floor_div, backend="turbine_cpu")
        result = opt_torch_scalar_convert(self.t)
        expected_result = floor_div(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")


if __name__ == '__main__':
    unittest.main()
