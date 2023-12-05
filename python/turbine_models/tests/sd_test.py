# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from turbine_models.custom_models.sd_inference import clip, unet, vae
import unittest
import os


class StableDiffusionTest(unittest.TestCase):
    def testExportClipModel(self):
        clip.export_clip_model(
            # This is a public model, so no auth required
            "CompVis/stable-diffusion-v1-4",
            None,
            "torch",
            "safetensors",
            "stable_diffusion_v1_4_clip.safetensors",
        )
        os.remove("stable_diffusion_v1_4_clip.safetensors")

    def testExportUnetModel(self):
        unet_model = unet.UnetModel(
            # This is a public model, so no auth required
            "CompVis/stable-diffusion-v1-4",
            None,
        )
        unet.export_unet_model(
            unet_model,
            # This is a public model, so no auth required
            "CompVis/stable-diffusion-v1-4",
            "torch",
            "safetensors",
            "stable_diffusion_v1_4_unet.safetensors",
        )
        os.remove("stable_diffusion_v1_4_unet.safetensors")

    def testExportUnetModel_v2_1(self):
        unet_model_v2_1 = unet.UnetModel(
            # This is a public model, so no auth required
            "stabilityai/stable-diffusion-2-1-base",
            None,
        )
        unet.export_unet_model(
            unet_model_v2_1,
            # This is a public model, so no auth required
            "stabilityai/stable-diffusion-2-1-base",
            "torch",
            "safetensors",
            "stable_diffusion_v2_1_unet.safetensors",
        )
        os.remove("stable_diffusion_v2_1_unet.safetensors")

    def testExportVaeModel(self):
        vae_model = vae.VaeModel(
            # This is a public model, so no auth required
            "CompVis/stable-diffusion-v1-4",
            None,
        )
        vae.export_vae_model(
            vae_model,
            # This is a public model, so no auth required
            "CompVis/stable-diffusion-v1-4",
            "torch",
            "safetensors",
            "stable_diffusion_v1_4_vae.safetensors",
        )
        os.remove("stable_diffusion_v1_4_vae.safetensors")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
