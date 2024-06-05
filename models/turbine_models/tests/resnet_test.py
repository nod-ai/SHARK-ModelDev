import argparse
import logging
from turbine_models.custom_models import resnet_18
import unittest
import os
import pytest

resnet_model = resnet_18.Resnet18Model()


class Resnet18Test(unittest.TestCase):
    def testExportResnet18ModelCPU(self):
        from turbine_models.tests.testing_cmd_opts import args
        arguments = {
            "run_vmfb": True,
            "compile_to": "vmfb",
            "vmfb_path": "",
            "device": "local-task",
            "target_triple": "x86_64-unknown-linux-gnu",
            "vulkan_max_allocation": "4294967296",
            "precision": "fp32",
        }
        resnet_18.export_resnet_18_model(
            resnet_model,
            "vmfb",
            "cpu",
        )
        namespace = AttributeDict(arguments)
        err = resnet_18.run_resnet_18_vmfb_comparison(resnet_model, namespace)
        assert err < 1e-5
        
    def testExportResnet18ModelStaticGFX1100(self):
        from turbine_models.tests.testing_cmd_opts import args
        arguments = {
            "run_vmfb": True,
            "compile_to": "vmfb",
            "vmfb_path": "",
            "device": "rocm",
            "target_triple": "gfx1100",
            "vulkan_max_allocation": "4294967296",
            "precision": "fp16",
        }
        resnet_18.export_static_resnet_18_model(
            resnet_model,
            "vmfb",
            "rocm",
            arguments["target_triple"],
        )
        namespace = AttributeDict(arguments)
        rocm_err = resnet_18.run_resnet_18_vmfb_comparison(resnet_model, namespace)
        namespace.device = "hip"
        hip_err = resnet_18.run_resnet_18_vmfb_comparison(resnet_model, namespace)
        print("ROCM ERROR:", rocm_err)
        print("HIP ERROR:", hip_err)
        assert rocm_err < 1e-5
        assert hip_err < 1e-5

    # def tearDown(self):
    #     if os.path.exists("resnet_18.vmfb"):
    #         os.remove("resnet_18.vmfb")
    #     if os.path.exists("resnet_18.mlir"):
    #         os.remove("resnet_18.mlir")


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
