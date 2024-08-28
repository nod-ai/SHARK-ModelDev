# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import copy
import torch
import math
import iree.runtime as ireert
from random import randint
from tqdm.auto import tqdm
from turbine_models.custom_models.sd_inference import (
    utils,
)
from turbine_models.custom_models.pipeline_base import (
    TurbinePipelineBase,
    merge_arg_into_map,
)
from turbine_models.custom_models.sd3_inference.text_encoder_impls import SDTokenizer
from turbine_models.custom_models.flux import (
    sampler,
    text_encoder,
    autoencoder,
    scheduler,
)
from turbine_models.model_runner import vmfbRunner
from transformers import T5Tokenizer, CLIPTokenizer
from pathlib import Path

from PIL import Image
import os
import numpy as np
import time
from datetime import datetime as dt
from einops import rearrange

# These are arguments common among submodel exports.
# They are expected to be populated in two steps:
# First, by the child class,
# and second by the base class for inference task-agnostic args.

flux_model_map = {
    "text_encoder": {
        "module_name": "compiled_text_encoder",
        "keywords": ["text_encoder"],
        "export_fn": text_encoder.export_text_encoders,
        "torch_module": text_encoder.TextEncoderModule,
        "use_metadata": False,
        "export_args": {
            "batch_size": 1,
            "max_length": 64,
            "decomp_attn": None,
        },
    },
    "sampler": {
        "module_name": "compiled_flux_sampler",
        "keywords": ["sampler"],
        "export_fn": sampler.export_flux_model,
        "torch_module": sampler.FluxModel,
        "use_metadata": True,
        "export_args": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024,
            "max_length": 64,
            "decomp_attn": None,
        },
    },
    "ae": {
        "module_name": "compiled_flux_a_e",
        "keywords": ["ae"],
        "dest_type": "numpy",
        "export_fn": autoencoder.export_ae_model,
        "torch_module": autoencoder.AEModel,
        "use_metadata": True,
        "export_args": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024,
            "num_channels": 16,
            "decomp_attn": None,
        },
    },
}


def get_sd_model_map(hf_model_name):
    return flux_model_map


torch_dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "int8": torch.int8,
    "i8": torch.int8,
}


class SharkFluxPipeline(TurbinePipelineBase):
    def __init__(
        self,
        hf_model_name: str | dict[str],
        height: int,
        width: int,
        batch_size: int,
        max_length: int | dict[int],
        precision: str | dict[str],
        device: str | dict[str],
        target: str | dict[str],
        ireec_flags: str | dict[str] = None,
        attn_spec: str | dict[str] = None,
        decomp_attn: bool | dict[bool] = False,
        pipeline_dir: str = "./shark_vmfbs",
        external_weights_dir: str = "./shark_weights",
        external_weights: str | dict[str] = "safetensors",
        num_inference_steps: int = 30,
        cpu_scheduling: bool = True,
        scheduler_id: str = None,  # compatibility only
        shift: float = 1.0,  # compatibility only
        use_i8_punet: bool = False,
        benchmark: bool | dict[bool] = False,
        verbose: bool = False,
        batch_prompts: bool = False,
        punet_quant_paths: dict[str] = None,
        vae_weight_path: str = None,
        vae_harness: bool = True,
        add_tk_kernels: bool = False,
        tk_kernels_dir: str | dict[str] = None,
        save_outputs: bool | dict[bool] = False,
    ):
        common_export_args = {
            "hf_model_name": None,
            "precision": None,
            "compile_to": "vmfb",
            "device": None,
            "target": None,
            "exit_on_vmfb": False,
            "pipeline_dir": pipeline_dir,
            "input_mlir": None,
            "ireec_flags": None,
            "attn_spec": attn_spec,
            "external_weights": None,
            "external_weight_path": None,
        }
        sd_model_map = copy.deepcopy(get_sd_model_map(hf_model_name))
        for submodel in sd_model_map:
            if "load" not in sd_model_map[submodel]:
                sd_model_map[submodel]["load"] = True
            sd_model_map[submodel]["export_args"]["batch_size"] = batch_size
            if "max_length" in sd_model_map[submodel]["export_args"]:
                max_length_sub = (
                    max_length if isinstance(max_length, int) else max_length[submodel]
                )
                sd_model_map[submodel]["export_args"]["max_length"] = max_length_sub
            if "height" in sd_model_map[submodel]["export_args"]:
                sd_model_map[submodel]["export_args"]["height"] = height
                sd_model_map[submodel]["export_args"]["width"] = width
            if "decomp_attn" in sd_model_map[submodel]["export_args"]:
                if isinstance(decomp_attn, bool):
                    sd_model_map[submodel]["export_args"]["decomp_attn"] = decomp_attn
                else:
                    sd_model_map[submodel]["export_args"]["decomp_attn"] = (
                        decomp_attn.get(submodel, False)
                    )
        super().__init__(
            sd_model_map,
            device,
            target,
            ireec_flags,
            precision,
            attn_spec,
            decomp_attn,
            external_weights,
            pipeline_dir,
            external_weights_dir,
            hf_model_name,
            benchmark,
            verbose,
            save_outputs,
            common_export_args,
        )
        for submodel in sd_model_map:
            if self.map[submodel].get("external_weights"):
                weights_filename = utils.create_safe_name(
                    self.map[submodel]["export_args"]["hf_model_name"],
                    f"_{submodel}_{self.map[submodel]['precision']}",
                )
                weights_filename += (
                    "." + self.map[submodel]["export_args"]["external_weights"]
                )
                self.map[submodel]["export_args"][
                    "external_weight_path"
                ] = weights_filename

        self.batch_size = batch_size
        self.model_max_length = self.max_length = max_length
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps

        self.text_encoder = None
        self.sampler = None
        self.ae = None
        self.scheduler = None

        self.split_scheduler = True
        self.add_tk_kernels = add_tk_kernels
        self.tk_kernels_dir = tk_kernels_dir

        self.base_model_name = (
            hf_model_name
            if isinstance(hf_model_name, str)
            else hf_model_name.get("sampler")
        )
        self.repo_name = (
            "black-forest-labs/FLUX.1-dev"
            if "schnell" not in self.base_model_name
            else "black-forest-labs/FLUX.1-schnell"
        )
        self.is_img2img = False

        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_tokenizer = SDTokenizer(tokenizer=clip_tokenizer)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            "google/t5-v1_1-xxl", max_length=self.model_max_length
        )
        self.scheduler_id = "FlowMatchEulerDiscrete"
        self.map["text_encoder"]["export_args"]["external_weights"] = "irpa"
        self.map["text_encoder"]["export_args"][
            "external_weight_path"
        ] = "flux_text_encoder_fp16.irpa"
        self.diffusion_model = self.map["sampler"]

        self.latents_precision = self.diffusion_model["precision"]
        if self.latents_precision == "bf16":
            self.latents_precision = "fp32"
        self.latents_channels = self.map["ae"]["export_args"]["num_channels"]
        self.scheduler_device = self.diffusion_model["device"]
        self.scheduler_driver = self.diffusion_model["driver"]
        self.scheduler_target = self.diffusion_model["target"]
        self.cast_latents_to_vae = False
        if self.diffusion_model["driver"] != self.map["ae"]["driver"]:
            self.cast_latents_to_vae = True
        self.latents_dtype = torch_dtypes[self.latents_precision]

    # LOAD

    def load_scheduler(
        self,
        scheduler_id: str = None,
        steps: int = 30,
    ):
        self.map["scheduler"] = {
            "module_name": "compiled_scheduler",
            "export_fn": scheduler.export_scheduler_model,
            "driver": self.scheduler_driver,
            "export_args": {
                "hf_model_name": self.repo_name,
                "scheduler_id": scheduler_id,
                "batch_size": self.batch_size,
                "height": self.height,
                "width": self.width,
                "num_inference_steps": steps,
                "precision": self.latents_precision,
                "compile_to": "vmfb",
                "device": self.scheduler_device,
                "target": self.scheduler_target,
                "ireec_flags": self.diffusion_model["export_args"]["ireec_flags"],
                "pipeline_dir": self.pipeline_dir,
            },
        }
        self.scheduler = None
        self.num_inference_steps = steps
        if scheduler_id:
            self.scheduler_id = scheduler_id
        scheduler_uid = "_".join(
            [
                f"{self.scheduler_id}Scheduler",
                f"bs{self.batch_size}",
                "x".join([str(self.width), str(self.height)]),
                self.latents_precision,
                str(self.num_inference_steps),
                self.scheduler_target,
            ]
        )
        scheduler_path = os.path.join(
            self.pipeline_dir,
            utils.create_safe_name(self.base_model_name, scheduler_uid) + ".vmfb",
        )
        if not os.path.exists(scheduler_path):
            self.export_submodel("scheduler")
        else:
            self.map["scheduler"]["vmfb"] = scheduler_path
        self.load_submodel("scheduler")

    # RUN

    def get_rand_latents(self, seed, num_samples=1):
        samples = []
        uint32_info = np.iinfo(np.uint32)
        uint32_min, uint32_max = uint32_info.min, uint32_info.max
        if seed < uint32_min or seed >= uint32_max:
            seed = randint(uint32_min, uint32_max)
        return torch.randn(
            num_samples,
            16,
            # allow for packing
            2 * math.ceil(self.height / 16),
            2 * math.ceil(self.width / 16),
            dtype=self.latents_dtype,
            generator=torch.Generator().manual_seed(int(seed)),
        )

    def prepare_latents(
        self,
        noise,
        num_inference_steps,
        image=None,
        strength=None,
    ):
        if self.is_img2img:
            raise NotImplementedError("Image-to-image not supported yet.")
        num_inference_steps = ireert.asdevicearray(
            self.scheduler.device, [num_inference_steps], dtype="int64"
        )
        return self.scheduler("run_initialize", [noise, num_inference_steps])

    def encode_prompt(self, text):
        t5_ids = self.t5_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).input_ids
        clip_ids = self.clip_tokenizer.tokenize_with_weights(text)
        text_encoder_inputs = [
            ireert.asdevicearray(self.text_encoder.device, t5_ids, dtype="int64"),
            ireert.asdevicearray(self.text_encoder.device, clip_ids, dtype="int64"),
        ]
        return self.text_encoder("encode_tokens", text_encoder_inputs)

    def produce_latents(
        self,
        sample,
        txt,
        txt_ids,
        vec,
        steps,
        guidance_scale,
    ):
        img, indexes, timesteps, img_ids = self.prepare_latents(sample, steps)
        guidance_vec = torch.full((sample.shape[0],), guidance_scale)
        guidance_scale = ireert.asdevicearray(
            self.sampler.device,
            guidance_vec,
            dtype=self.diffusion_model["np_dtype"],
        )
        timesteps_cpu = timesteps
        timesteps_list_gpu = [
            ireert.asdevicearray(
                self.scheduler.device,
                [timesteps_cpu[i]],
                dtype=self.diffusion_model["np_dtype"],
            )
            for i in range(steps)
        ]
        for t_curr, t_prev in zip(timesteps_list_gpu[:-1], timesteps_list_gpu[1:]):
            sampler_inputs = [
                img,
                img_ids,
                txt,
                txt_ids,
                vec,
                t_curr,
                t_prev,
                guidance_scale,
            ]
            pred = self.sampler(
                "run_forward",
                sampler_inputs,
            )
            sampler_inputs[0] = sampler_inputs[0] + (t_prev.to_host() - t_curr.to_host()) * pred
        return sampler_inputs[0]

    def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 30,
        batch_count: int = 1,
        guidance_scale: float = 7.5,
        seed: float = -1,
        cpu_scheduling: bool = True,
        scheduler_id: str = "EulerDiscrete",
        return_imgs: bool = False,
        seed_increment: int = 1,
    ):
        needs_new_scheduler = (
            (steps and steps != self.num_inference_steps)
            or (cpu_scheduling != self.cpu_scheduling)
            and self.split_scheduler
        )
        if not self.scheduler and not self.compiled_pipeline:
            needs_new_scheduler = True

        if guidance_scale == 0:
            negative_prompt = prompt
            prompt = ""

        self.cpu_scheduling = cpu_scheduling
        if steps and needs_new_scheduler:
            self.num_inference_steps = steps
            self.load_scheduler(scheduler_id, steps)

        pipe_start = time.time()
        numpy_images = []
        samples = []
        for i in range(batch_count):
            samples.extend([self.get_rand_latents(seed, self.batch_size)])
            seed += seed_increment
        txt, vec = self.encode_prompt(prompt)
        t_ids = torch.zeros(self.batch_size, self.max_length, 3)
        txt_ids = ireert.asdevicearray(
            self.sampler.device, t_ids, dtype=self.diffusion_model["np_dtype"]
        )
        for i in range(batch_count):
            produce_latents_input = [
                samples[i],
                txt,
                txt_ids,
                vec,
                steps,
                guidance_scale,
            ]
            latents = self.produce_latents(*produce_latents_input)

            if self.cast_latents_to_vae:
                latents = ireert.asdevicearray(
                    self.ae.device,
                    latents.to_host(),
                    dtype=self.map["ae"]["np_dtype"],
                )
            breakpoint()
            image = self.ae("decode", [latents])
            numpy_images.append(image)
            pipe_end = time.time()

        logging.info(f"Total inference time: {pipe_end - pipe_start:.2f}s")
        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        images = []
        for idx, image in enumerate(numpy_images):
            images.append(Image.fromarray(image.astype("uint8")))
        if return_imgs:
            return images
        for idx, image in enumerate(images):
            img_path = "flux_output_" + timestamp + "_" + str(idx) + ".png"
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
    from turbine_models.custom_models.sd_inference.sd_cmd_opts import args

    ireec_flags = {
        "text_encoder": args.ireec_flags + args.clip_flags,
        "scheduler": args.ireec_flags,
        "sampler": args.ireec_flags + args.sampler_flags,
        "ae": args.ireec_flags + args.vae_flags,
    }
    devices = {
        "text_encoder": args.clip_device if args.clip_device else args.device,
        "scheduler": args.scheduler_device if args.scheduler_device else args.device,
        "sampler": args.sampler_device if args.sampler_device else args.device,
        "ae": args.vae_device if args.vae_device else args.device,
    }
    targets = {
        "text_encoder": (
            args.clip_target if args.clip_target else args.iree_target_triple
        ),
        "scheduler": (
            args.scheduler_target if args.scheduler_target else args.iree_target_triple
        ),
        "sampler": (
            args.sampler_target if args.sampler_target else args.iree_target_triple
        ),
        "ae": args.vae_target if args.vae_target else args.iree_target_triple,
    }
    precisions = {
        "text_encoder": args.clip_precision if args.clip_precision else args.precision,
        "sampler": args.sampler_precision if args.sampler_precision else args.precision,
        "ae": args.vae_precision if args.vae_precision else args.precision,
    }
    specs = {
        "text_encoder": args.clip_spec if args.clip_spec else args.attn_spec,
        "sampler": args.sampler_spec if args.sampler_spec else args.attn_spec,
        "ae": args.vae_spec if args.vae_spec else args.attn_spec,
    }
    if not args.pipeline_dir:
        args.pipeline_dir = utils.create_safe_name(args.hf_model_name, "")
    benchmark = {}
    if args.benchmark:
        if args.benchmark.lower() == "all":
            benchmark = True
        else:
            for i in args.benchmark.split(","):
                benchmark[i] = True
    else:
        benchmark = False
    if args.save_outputs:
        if args.save_outputs.lower() == "all":
            save_outputs = True
        else:
            for i in args.save_outputs.split(","):
                save_outputs[i] = True
    else:
        save_outputs = False
    args.decomp_attn = {
        "text_encoder": (
            args.clip_decomp_attn if args.clip_decomp_attn else args.decomp_attn
        ),
        "sampler": (
            args.sampler_decomp_attn if args.sampler_decomp_attn else args.decomp_attn
        ),
        "ae": args.vae_decomp_attn if args.vae_decomp_attn else args.decomp_attn,
    }
    flux_pipe = SharkFluxPipeline(
        args.hf_model_name,
        args.height,
        args.width,
        args.batch_size,
        args.max_length,
        precisions,
        devices,
        targets,
        ireec_flags,
        specs,
        args.decomp_attn,
        args.pipeline_dir,
        args.external_weights_dir,
        args.external_weights,
        args.num_inference_steps,
        args.cpu_scheduling,
        args.scheduler_id,
        None,
        args.use_i8_punet,
        benchmark,
        args.verbose,
        save_outputs=save_outputs,
    )
    flux_pipe.prepare_all()
    flux_pipe.load_scheduler("FlowMatchEulerDiscrete", args.num_inference_steps)
    flux_pipe.load_map()
    flux_pipe.generate_images(
        args.prompt,
        args.negative_prompt,
        args.num_inference_steps,
        args.batch_count,
        args.guidance_scale,
        args.seed,
        args.cpu_scheduling,
        args.scheduler_id,
        False,
    )
    print("Image generation complete.")
