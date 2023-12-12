# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import logging
from turbine_models.custom_models.sd_inference import clip, unet, vae
import unittest
import os


arguments = {
    "hf_auth_token": None,
    "hf_model_name": "CompVis/stable-diffusion-v1-4",
    "batch_size": 1,
    "height": 512,
    "width": 512,
    "run_vmfb": True,
    "compile_to": None,
    "external_weight_file": "",
    "vmfb_path": "",
    "external_weights": None,
    "device": "local-task",
    "iree_target_triple": "",
    "vulkan_max_allocation": "4294967296",
}


unet_model = unet.UnetModel(
    # This is a public model, so no auth required
    "CompVis/stable-diffusion-v1-4",
    None,
)

vae_model = vae.VaeModel(
    # This is a public model, so no auth required
    "CompVis/stable-diffusion-v1-4",
    None,
)


class StableDiffusionTest(unittest.TestCase):
    def testExportClipModel(self):
        with self.assertRaises(SystemExit) as cm:
            clip.export_clip_model(
                # This is a public model, so no auth required
                "CompVis/stable-diffusion-v1-4",
                None,
                "vmfb",
                "safetensors",
                "stable_diffusion_v1_4_clip.safetensors",
                "cpu",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_file"] = "stable_diffusion_v1_4_clip.safetensors"
        namespace = argparse.Namespace(**arguments)
        clip.run_clip_vmfb_comparison(namespace)
        os.remove("stable_diffusion_v1_4_clip.safetensors")
        os.remove("stable_diffusion_v1_4_clip.vmfb")

    def testExportUnetModel(self):
        with self.assertRaises(SystemExit) as cm:
            unet.export_unet_model(
                unet_model,
                # This is a public model, so no auth required
                "CompVis/stable-diffusion-v1-4",
                arguments["batch_size"],
                arguments["height"],
                arguments["width"],
                None,
                "vmfb",
                "safetensors",
                "stable_diffusion_v1_4_unet.safetensors",
                "cpu",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_file"] = "stable_diffusion_v1_4_unet.safetensors"
        namespace = argparse.Namespace(**arguments)
        unet.run_unet_vmfb_comparison(unet_model, namespace)
        os.remove("stable_diffusion_v1_4_unet.safetensors")
        os.remove("stable_diffusion_v1_4_unet.vmfb")

    def testExportVaeModel(self):
        with self.assertRaises(SystemExit) as cm:
            vae.export_vae_model(
                vae_model,
                # This is a public model, so no auth required
                "CompVis/stable-diffusion-v1-4",
                arguments["batch_size"],
                arguments["height"],
                arguments["width"],
                None,
                "vmfb",
                "safetensors",
                "stable_diffusion_v1_4_vae.safetensors",
                "cpu",
            )
        self.assertEqual(cm.exception.code, None)
        arguments["external_weight_file"] = "stable_diffusion_v1_4_vae.safetensors"
        namespace = argparse.Namespace(**arguments)
        vae.run_vae_vmfb_comparison(vae_model, namespace)
        os.remove("stable_diffusion_v1_4_vae.safetensors")
        os.remove("stable_diffusion_v1_4_vae.vmfb")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
