# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from shark_turbine.aot import ExternalTensorTrait
from turbine_llm.types import *


def _t(name: str, *dims: int):
    return DefaultPrimitiveTensor(name, torch.empty(*dims))


def _flat_t_dict(*ts):
    return {t.name: t for t in ts}


class ThetaTest(unittest.TestCase):
    def testThetaAccess(self):
        # TODO: Make construction of a Theta programatically more natural.
        theta = Theta(
            _flat_t_dict(
                _t("a.b.c", 1, 2),
                _t("a.c.d", 10, 11),
                _t("1.2.3", 3, 4),
            )
        )

        # No root tensors.
        self.assertFalse(theta.tensors)

        flattened = theta.flatten()
        self.assertIn("a.b.c", flattened)
        self.assertIn("a.c.d", flattened)
        self.assertIn("1.2.3", flattened)

        print(theta.keys)
        self.assertIn("a", theta.keys)
        self.assertIn("1", theta.keys)

        sub_theta = theta("a")
        self.assertIn("b", sub_theta.keys)
        self.assertIn("c", sub_theta.keys)
        self.assertEqual("a.b.c", sub_theta.tensor("b", "c").name)
        self.assertEqual("a.c.d", sub_theta.tensor("c", "d").name)

        sub_sub_theta = theta("a", "b")
        self.assertEqual("a.b.c", sub_sub_theta.tensors[0].name)


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp("_dstest"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def testDatasetRoundtrip(self):
        theta = Theta(
            _flat_t_dict(
                _t("a.b.c", 1, 2),
                _t("a.c.d", 10, 11),
                _t("1.2.3", 3, 4),
            )
        )

        ds_src = Dataset({"foo": "bar"}, theta)
        ds_src.save(self.temp_dir / "myds.irpa")

        # Not mmap'ing makes things a bit better on Windows for tests.
        ds_load = Dataset.load(self.temp_dir / "myds.irpa", mmap=False)
        self.assertEqual("bar", ds_load.properties["foo"])
        t_abc = ds_load.root_theta.tensor("a", "b", "c")
        t_acd = ds_load.root_theta.tensor("a", "c", "d")
        t_123 = ds_load.root_theta.tensor("1", "2", "3")
        self.assertEqual([1, 2], list(t_abc.shape))
        self.assertEqual([10, 11], list(t_acd.shape))
        self.assertEqual([3, 4], list(t_123.shape))
        self.assertEqual(
            "a.b.c", ExternalTensorTrait.get(t_abc.as_torch()).external_name
        )
        self.assertEqual(
            "a.c.d", ExternalTensorTrait.get(t_acd.as_torch()).external_name
        )

    def _createTestLayout(self):
        n = 128
        k = 1024
        bs = 32

        return BlockScaledLayout(
            [n, k],
            d=torch.empty(n, k // bs, 1, dtype=torch.float32),
            qs=torch.empty(n, k // bs, bs, dtype=torch.int8),
            m=torch.empty(n, k // bs, bs, dtype=torch.float32),
        )

    def testRoundtripPlanarQuantizedTensor(self):
        layout_in = self._createTestLayout()
        t_orig = PlanarQuantizedTensor("a.b.c", shape=layout_in.shape, layout=layout_in)
        self.assertIs(t_orig, t_orig.to_planar())
        ds_orig = Dataset({}, Theta({t_orig.name: t_orig}))
        ds_orig.save(self.temp_dir / "myds.irpa")

        ds_load = Dataset.load(self.temp_dir / "myds.irpa", mmap=False)
        t_load = ds_load.root_theta.tensor("a", "b", "c")
        print(t_load)
        self.assertEqual(t_load.shape, t_orig.shape)
        self.assertEqual(t_load.globals.keys(), t_orig.globals.keys())
        self.assertEqual(t_load.layout.shape, t_orig.layout.shape)
        self.assertEqual(t_load.layout.planes.keys(), t_orig.layout.planes.keys())
        self.assertEqual(t_load.layout.metadata, t_orig.layout.metadata)


if __name__ == "__main__":
    unittest.main()
