# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import torch.nn as nn

from iree.runtime import (
    ParameterIndex,
    ParameterProvider,
)

from shark_turbine.aot import (
    export,
)

from shark_turbine.aot.params import (
    externalize_module_parameters,
    save_module_parameters,
    ExternalTensor,
    ParameterArchive,
)


class SimpleParamsModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(20, 30)

    def forward(self, x):
        return self.classifier(x)


class ParamsTest(unittest.TestCase):
    def testCreateArchive(self):
        file_path = "/tmp/mystuff.irpa"
        m = SimpleParamsModule()
        save_module_parameters(file_path, m)
        archive = ParameterArchive(file_path)
        print(archive)

    def testExportExternalized(self):
        m = SimpleParamsModule()
        externalize_module_parameters(m)
        export(m, args=(torch.empty([128, 20]),), global_params=False).print_readable()


class ExternalTensorTest(unittest.TestCase):

    def testBackedExternalTensor(self):
        inner_t = torch.ones([2, 3], dtype=torch.float32)
        t = ExternalTensor(
            inner_t, requires_grad=True, external_name="foobar", external_scope="main"
        )
        self.assertIs(type(t), ExternalTensor)
        self.assertTrue(t._is_turbine_external_tensor)
        self.assertEqual("main", t.external_scope)
        self.assertEqual("foobar", t.external_name)
        r = t + 1
        self.assertFalse(hasattr(r, "_is_turbine_external_tensor"))
        self.assertFalse(hasattr(r, "external_name"))
        self.assertFalse(hasattr(r, "external_scope"))

    def testParameter(self):
        p = nn.Parameter(torch.ones([2, 3], dtype=torch.float32))
        t = ExternalTensor(p, external_name="foobar", external_scope="main")
        self.assertIs(type(t), ExternalTensor)
        self.assertTrue(t._is_turbine_external_tensor)
        self.assertEqual("main", t.external_scope)
        self.assertEqual("foobar", t.external_name)
        self.assertTrue(isinstance(t, nn.Parameter))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
