# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import copy
import torch
import iree.runtime as ireert
from turbine_models.custom_models.sd_inference import (
    clip,
    unet,
    vae,
    schedulers,
    utils,
)
from turbine_models.model_runner import vmfbRunner
from transformers import CLIPTokenizer

from PIL import Image
import os
import numpy as np
import time
from datetime import datetime as dt

device_list = [
    "cpu",
    "vulkan",
    "cuda",
    "rocm",
]

rt_device_list = [
    "local-task",
    "local-sync",
    "vulkan",
    "cuda",
    "rocm",
    "hip",
]

SUBMODELS = {
    "clip": None,
    "scheduler": None,
    "unet": None,
    "vae_decode": None,
}

class SharkSDPipeline:
    def __init__(
        self,
        hf_model_name: str,
        scheduler_id: str,
        height: int,
        width: int,
        precision: str,
        max_length: int,
        batch_size: int,
        num_inference_steps: int,
        device: str,
        iree_target_triple: str,
        ireec_flags: dict = EMPTY_FLAGS,
        attn_spec: str = None,
        decomp_attn: bool = False,
        pipeline_dir: str = "./shark_vmfbs",
        external_weights_dir: str = "./shark_weights",
        external_weights: str = "safetensors",
        vae_decomp_attn: bool = True,
    ):
        self.hf_model_name = hf_model_name
        self.scheduler_id = scheduler_id
        self.height = height
        self.width = width
        self.precision = precision
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.iree_target_triple = iree_target_triple
        self.ireec_flags = ireec_flags if ireec_flags else copy.deepcopy(SUBMODELS)
        self.attn_spec = attn_spec
        self.decomp_attn = decomp_attn
        self.pipeline_dir = pipeline_dir
        self.external_weights_dir = external_weights_dir
        self.external_weights = external_weights
        self.vae_decomp_attn = vae_decomp_attn
        self.is_sdxl = "xl" in self.hf_model_name

    # FILE MANAGEMENT AND PIPELINE SETUP

    def check_prepared(
        self,
        mlirs: dict,
        vmfbs: dict,
        weights: dict,
        interactive: bool = True,
    ):
        ready, vmfbs, weights = self.is_prepared(vmfbs, weights)
        if not ready:
            if interactive:
                do_continue = input(
                    f"\nIt seems you are missing some necessary files. Would you like to generate them now? (y/n)"
                )
                if do_continue.lower() != "y":
                    exit()
            else:
                do_continue = "y"
            if do_continue.lower() == "y":
                for submodel in vmfbs.keys():
                    if vmfbs[submodel] == None:
                        vmfb, weight = self.export_submodel(submodel, input_mlir=mlirs)
                        vmfbs[submodel] = vmfb
                        if weights[submodel] is None:
                            weights[submodel] = weight
                    elif weights[submodel] is None and "scheduler" not in submodel:
                        _, weight = self.export_submodel(submodel, weights_only=True)
                        weights[submodel] = weight
                ready, vmfbs, weights = self.is_prepared(vmfbs, weights)
                if ready:
                    print("All necessary files found. Generating images.")
                    return vmfbs, weights
                else:
                    print("There was an error generating the necessary files.")
                    exit()
        else:
            print("All necessary files found. Loading pipeline.")
        return vmfbs, weights

    def is_prepared(self, vmfbs, weights):
        missing = []
        for key in vmfbs:
            default_filepath = os.path.join(self.pipeline_dir, key + "_" + self.iree_target_triple + ".vmfb")
            if vmfbs[key] is not None and os.path.exists(vmfbs[key]):
                continue
            elif vmfbs[key] == None and os.path.exists(default_filepath):
                vmfbs[key] = default_filepath
            else:
                missing.append(key + ".vmfb")
        for w_key in weights:
            if "scheduler" in w_key:
                continue
            if weights[w_key] is not None and os.path.exists(weights[w_key]):
                continue
            default_name = os.path.join(
                self.external_weights_dir, w_key + "." + self.external_weights
            )
            if weights[w_key] is None and os.path.exists(default_name):
                weights[w_key] = os.path.join(default_name)
            else:
                missing.append(w_key + "." + self.external_weights)
        if len(missing) > 0:
            print(f"Missing files: " + ", ".join(missing))
            return False, vmfbs, weights
        else:
            return True, vmfbs, weights

    def get_mlir_from_turbine_tank(self, submodel, container_name):
        from turbine_models.turbine_tank import downloadModelArtifacts

        safe_name = utils.create_safe_name(
            self.hf_model_name,
            f"_{self.max_length}_{self.height}x{self.width}_{self.precision}_{submodel}.mlir",
        )
        mlir_path = downloadModelArtifacts(
            safe_name,
            container_name,
        )
        return mlir_path

    # IMPORT / COMPILE PHASE

    def get_torch_models(self, submodel):
        match submodel:
            case "unet":
                unet_torch = unet.UnetModel(
                    self.hf_model_name,
                    self.height,
                    self.width,
                    self.batch_size,
                    None,
                    precision=self.precision,
                )
                return unet_torch
            case "vae_decode":
                if not self.custom_vae:
                    custom_vae = "madebyollin/sdxl-vae-fp16-fix" if self.precision == "fp16" and self.is_sdxl else None
                vae_torch = vae.VaeModel(
                    self.hf_model_name,
                    custom_vae,
                )
                return vae_torch

    def export_submodel(
        self,
        submodel: str,
        input_mlir: str = None,
        weights_only: bool = False,
    ):
        if not os.path.exists(self.pipeline_dir):
            os.makedirs(self.pipeline_dir)
        if self.external_weights_dir:
            if not os.path.exists(self.external_weights_dir):
                os.makedirs(external_weights_dir, exist_ok=True)
            vae_external_weight_path = os.path.join(
                self.external_weights_dir, "vae_decode." + self.external_weights
            )
            unet_external_weight_path = os.path.join(
                self.external_weights_dir, "unet." + self.external_weights
            )
            clip_external_weight_path = os.path.join(
                self.external_weights_dir, "clip." + self.external_weights
            )
        elif self.external_weights is None:
            print(
                "No external weights type specified using --external_weights, weights for imported .mlir files will not be externalized."
            )
            vae_external_weight_path = None
            unet_external_weight_path = None
            clip_external_weight_path = None
        else:
            print(
                f"No external weights directory specified using --external_weights_dir, we assume you have your own weights in {self.pipeline_dir}."
            )
            external_weights_dir = self.pipeline_dir
            if not os.path.exists(self.pipeline_dir):
                os.makedirs(self.pipeline_dir, exist_ok=True)
            vae_external_weight_path = os.path.join(
                self.pipeline_dir, "vae_decode." + self.external_weights
            )
            unet_external_weight_path = os.path.join(
                self.pipeline_dir, "unet." + self.external_weights
            )
            clip_external_weight_path = os.path.join(
                self.pipeline_dir, "clip." + self.external_weights
            )
        if weights_only:
            input_mlir = copy.deepcopy(SUBMODELS)
        match submodel:
            case "clip":
                _, clip_vmfb = clip.export_combined_clip(
                    self.hf_model_name,
                    None,
                    self.max_length,
                    self.precision,
                    "vmfb",
                    self.external_weights,
                    clip_external_weight_path,
                    self.device,
                    self.iree_target_triple,
                    self.ireec_flags["clip"],
                    exit_on_vmfb=False,
                    pipeline_dir=self.pipeline_dir,
                    input_mlir=input_mlir["clip"],
                    attn_spec=self.attn_spec,
                    weights_only=weights_only,
                )
                return clip_vmfb, clip_external_weight_path
            case "scheduler":
                scheduler_vmfb = schedulers.export_scheduler(
                    self.hf_model_name,
                    self.scheduler_id,
                    self.batch_size,
                    self.height,
                    self.width,
                    self.num_inference_steps,
                    self.precision,
                    "vmfb",
                    self.device,
                    self.iree_target_triple,
                    self.ireec_flags["scheduler"],
                    exit_on_vmfb=False,
                    pipeline_dir=self.pipeline_dir,
                    input_mlir=input_mlir["scheduler"],
                )
                return scheduler_vmfb, None
            case "unet":
                if input_mlir[submodel]:
                    unet_torch = None
                else:
                    unet_torch = self.get_torch_models("unet")
                    
                unet_vmfb = unet.export_unet_model(
                    unet_torch,
                    self.hf_model_name,
                    self.batch_size,
                    self.height,
                    self.width,
                    self.precision,
                    self.max_length,
                    None,
                    "vmfb",
                    self.external_weights,
                    unet_external_weight_path,
                    self.device,
                    self.iree_target_triple,
                    self.ireec_flags["unet"],
                    self.decomp_attn,
                    exit_on_vmfb=False,
                    pipeline_dir=self.pipeline_dir,
                    attn_spec=self.attn_spec,
                    input_mlir=input_mlir["unet"],
                    weights_only=weights_only,
                )
                return unet_vmfb, unet_external_weight_path
            case "vae_decode":
                if not input_mlir[submodel]:
                    vae_torch = self.get_torch_models("vae_decode")
                else:
                    vae_torch = None
                vae_decode_vmfb = vae.export_vae_model(
                    vae_torch,
                    self.hf_model_name,
                    self.batch_size,
                    self.height,
                    self.width,
                    self.precision,
                    "vmfb",
                    self.external_weights,
                    vae_external_weight_path,
                    self.device,
                    self.iree_target_triple,
                    self.ireec_flags["vae"],
                    "decode",
                    self.vae_decomp_attn,
                    exit_on_vmfb=False,
                    pipeline_dir=self.pipeline_dir,
                    attn_spec=self.attn_spec,
                    input_mlir=input_mlir["vae_decode"],
                    weights_only=weights_only,
                )
                return vae_decode_vmfb, vae_external_weight_path


    # LOAD

    def load_pipeline(
        self,
        vmfbs: dict,
        weights: dict,
        rt_device: str = "local-task",
        compiled_pipeline: bool = False,
    ):
        self.runners = {}
        runners = {}
        runners["tokenizers"] = []
        runners["tokenizers"] += CLIPTokenizer.from_pretrained(
            self.hf_model_name,
            subfolder="tokenizer",
        )
        if self.is_sdxl:
            runners["tokenizers"] += CLIPTokenizer.from_pretrained(
                self.hf_model_name,
                subfolder="tokenizer_2",
            ),
            
        runners["clip"] = vmfbRunner(
            rt_device, vmfbs["clip"], weights["clip"]
        )
        runners["scheduler"] = vmfbRunner(
            rt_device, vmfbs["scheduler"], weights["scheduler"]
        )
        runners["unet"] = vmfbRunner(
            rt_device, vmfbs["unet"], weights["unet"]
        )
        runners["vae_decode"] = vmfbRunner(
            rt_device, vmfbs["vae_decode"], weights["vae_decode"]
        )
        self.runners = runners
        self.compiled_pipeline = False
        print("Successfully loaded pipeline.")

    # RUN

    def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        batch_count: int = 1,
        guidance_scale: float = 7.5,
        seed: float = -1,
        return_imgs: bool = False,
    ):
        # TODO: implement case where this is false e.g. in SDXL Turbo
        # do_classifier_free_guidance = True

        iree_dtype = "float32" if self.precision == "fp32" else "float16"
        torch_dtype = torch.float32 if self.precision == "fp32" else torch.float16

        pipe_start = time.time()

        max_length = self.max_length

        samples = []
        numpy_images = []


        for i in range(batch_count):
            generator = torch.random.manual_seed(seed + i)
            rand_sample = torch.randn(
                (
                    self.batch_size,
                    4,
                    self.height // 8,
                    self.width // 8,
                ),
                generator=generator,
                dtype=torch_dtype,
            )
            samples.append(
                ireert.asdevicearray(
                    self.runners["unet"].config.device, rand_sample, dtype=iree_dtype
                )
            )

        guidance_scale = ireert.asdevicearray(
            self.runners["unet"].config.device,
            np.asarray([guidance_scale]),
            dtype=iree_dtype,
        )

        text_input_ids_list = []
        uncond_input_ids_list = []

        tokenize_start = time.time()

        # Tokenize prompt and negative prompt.
        for tokenizer in self.runners["tokenizers"]:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            uncond_input_ids = uncond_input.input_ids

            text_input_ids_list.extend(
                [
                    ireert.asdevicearray(
                        self.runners["unet"].config.device, text_input_ids
                    )
                ]
            )
            uncond_input_ids_list.extend(
                [
                    ireert.asdevicearray(
                        self.runners["unet"].config.device, uncond_input_ids
                    )
                ]
            )
        encode_prompts_start = time.time()

        prompt_embeds, add_text_embeds = self.runners[
            "clip"
        ].ctx.modules.compiled_clip["encode_prompts"](
            *text_input_ids_list, *uncond_input_ids_list
        )

        encode_prompts_end = time.time()

        for i in range(batch_count):
            unet_start = time.time()

            sample, add_time_ids, timesteps = self.runners["scheduler"].ctx.modules.scheduler[
                "init"
            ](samples[i])

            for t in range(timesteps):
                latents = self.runners["scheduler"].ctx.modules.scheduler["scale"](
                    sample, t
                )
                latents = self.runners["unet"].ctx.modules.compiled_unet["main"](
                    latents, prompt_embeds, add_text_embeds, add_time_ids, guidance_scale, t
                )
                sample = self.runners["scheduler"].ctx.modules.scheduler["step"](
                    sample, latents, t
                )
            
            vae_start = time.time()
            vae_out = self.runners["vae_decode"].ctx.modules.compiled_vae["main"](
                sample
            )

            pipe_end = time.time()

            image = vae_out.to_host()

            numpy_images.append(image)
            print("Batch #", i + 1, "\n")
            print(
                "UNet time(",
                self.num_inference_steps,
                "): ",
                vae_start - unet_start,
                "sec,",
            )
            print(
                "Unet average step latency: ",
                (vae_start - unet_start) / self.num_inference_steps,
                "sec",
            )
            print("VAE time: ", pipe_end - vae_start, "sec")
            print(
                f"\nTotal time (txt2img, batch #{str(i+1)}): ",
                (encode_prompts_end - encode_prompts_start)
                + (pipe_end - unet_start),
                "sec\n",
            )
        end = time.time()
        print("Total CLIP time:", encode_prompts_end - encode_prompts_start, "sec")
        print("Total tokenize time:", encode_prompts_start - tokenize_start, "sec")
        print("Loading time: ", encode_prompts_start - pipe_start, "sec")
        if batch_count > 1:
            print(
                f"Total inference time ({batch_count} batch(es)):",
                end - encode_prompts_start,
                "sec",
            )
        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        images = []
        for idx, image in enumerate(numpy_images):
            image = torch.from_numpy(image).cpu().permute(0, 2, 3, 1).float().numpy()
            image = numpy_to_pil_image(image)
            images.append(image[0])
        if return_imgs:
            return images
        for idx, image in enumerate(images):
            img_path = "sdxl_output_" + timestamp + "_" + str(idx) + ".png"
            image.save(img_path)
            print(img_path, "saved")
        return


def numpy_to_pil_image(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    mlirs = copy.deepcopy(SUBMODELS)
    vmfbs = copy.deepcopy(SUBMODELS)
    weights = copy.deepcopy(SUBMODELS)
    ireec_flags = {
        "clip": args.ireec_flags + args.clip_flags,
        "scheduler": args.ireec_flags,
        "unet": args.ireec_flags + args.unet_flags,
        "vae_decode": args.ireec_flags + args.vae_flags,
    }
    if not args.pipeline_dir:
        pipe_id_list = [
            utils.create_safe_name(args.hf_model_name, args.iree_target_triple),
            str(args.height),
            str(args.width),
            str(args.max_length),
            args.precision,
            args.device,
        ]
        args.pipeline_dir = os.path.join(
            ".",
            "_".join(pipe_id_list),
        )
    if args.input_mlir:
        user_mlir_list = args.input_mlir.split(",")
    else:
        user_mlir_list = []
    for submodel_id, mlir_path in zip(mlirs.keys(), user_mlir_list):
        if submodel_id in mlir_path:
            mlirs[submodel_id] = mlir_path
    if not args.external_weights_dir and args.external_weights:
        args.external_weights_dir = args.pipeline_dir

    sd_pipe = SharkSDPipeline(
        args.hf_model_name,
        args.scheduler_id,
        args.height,
        args.width,
        args.precision,
        args.max_length,
        args.batch_size,
        args.num_inference_steps,
        args.device,
        args.iree_target_triple,
        ireec_flags,
        args.attn_spec,
        args.decomp_attn,
        args.pipeline_dir,
        args.external_weights_dir,
        args.external_weights,
        args.vae_decomp_attn,
    )
    vmfbs, weights = sd_pipe.check_prepared(mlirs, vmfbs, weights)
    sd_pipe.load_pipeline(vmfbs, weights, args.rt_device, args.compiled_pipeline)
    sd_pipe.generate_images(
        args.prompt,
        args.negative_prompt,
        args.batch_count,
        args.guidance_scale,
        args.seed,
        False,
    )
    print("Image generation complete.")