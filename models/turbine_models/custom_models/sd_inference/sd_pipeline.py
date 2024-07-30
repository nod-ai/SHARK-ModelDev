# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import copy
import torch
import iree.runtime as ireert
from random import randint
from tqdm.auto import tqdm
from turbine_models.custom_models.sd_inference import (
    clip,
    unet,
    vae,
    schedulers,
    utils,
)
from turbine_models.custom_models.sdxl_inference import (
    sdxl_prompt_encoder as sdxl_clip,
    unet as sdxl_unet,
)
from turbine_models.custom_models.sd3_inference import (
    sd3_text_encoders,
    sd3_mmdit,
    sd3_schedulers,
)
from turbine_models.custom_models.sd3_inference.text_encoder_impls import SD3Tokenizer
from turbine_models.custom_models.pipeline_base import (
    TurbinePipelineBase,
    merge_arg_into_map,
)
from turbine_models.custom_models.sd_inference.tokenization import encode_prompt
from turbine_models.model_runner import vmfbRunner
from transformers import CLIPTokenizer
from pathlib import Path

from PIL import Image
import os
import numpy as np
import time
from datetime import datetime as dt

# These are arguments common among submodel exports.
# They are expected to be populated in two steps:
# First, by the child class,
# and second by the base class for inference task-agnostic args.

sd1_sd2_model_map = {
    "text_encoder": {
        "module_name": "compiled_text_encoder",
        "keywords": ["clip"],
        "dest_type": "torch",
        "export_fn": clip.export_clip_model,
        "export_args": {
            "batch_size": 1,
            "max_length": 64,
        },
    },
    "unet": {
        "module_name": "compiled_unet",
        "keywords": ["unet"],
        "export_fn": unet.export_unet_model,
        "export_args": {
            "batch_size": 1,
            "height": 512,
            "width": 512,
            "max_length": 64,
            "decomp_attn": None,
        },
    },
    "vae": {
        "module_name": "compiled_vae",
        "keywords": ["vae"],
        "dest_type": "numpy",
        "export_fn": vae.export_vae_model,
        "export_args": {
            "batch_size": 1,
            "height": 512,
            "width": 512,
            "num_channels": 4,
            "decomp_attn": None,
        },
    },
}
sdxl_model_map = {
    "text_encoder": {
        "module_name": "compiled_clip",
        "keywords": ["prompt_encoder"],
        "dest_type": "torch",
        "export_fn": sdxl_clip.export_prompt_encoder,
        "export_args": {
            "batch_size": 1,
            "max_length": 64,
        },
    },
    "unet": {
        "module_name": "compiled_unet",
        "keywords": ["unet", "!loop"],
        "export_fn": sdxl_unet.export_unet_model,
        "export_args": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024,
            "max_length": 64,
            "decomp_attn": None,
        },
    },
    "vae": {
        "module_name": "compiled_vae",
        "keywords": ["vae"],
        "dest_type": "numpy",
        "export_fn": vae.export_vae_model,
        "export_args": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024,
            "num_channels": 4,
            "decomp_attn": None,
        },
    },
    "unetloop": {
        "module_name": "sdxl_compiled_pipeline",
        "load": False,
        "keywords": ["unetloop"],
        "wraps": ["unet", "scheduler"],
        "export_args": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024,
            "max_length": 64,
        },
    },
    "fullpipeline": {
        "module_name": "sdxl_compiled_pipeline",
        "load": False,
        "keywords": ["fullpipeline"],
        "wraps": ["text_encoder", "unet", "scheduler", "vae"],
        "export_args": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024,
            "max_length": 64,
        },
    },
}
sd3_model_map = {
    "text_encoder": {
        "module_name": "compiled_text_encoder",
        "keywords": ["text_encoder"],
        "export_fn": sd3_text_encoders.export_text_encoders,
        "export_args": {
            "batch_size": 1,
            "max_length": 64,
        },
    },
    "mmdit": {
        "module_name": "compiled_mmdit",
        "keywords": ["mmdit"],
        "export_fn": sd3_mmdit.export_mmdit_model,
        "export_args": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024,
            "max_length": 64,
            "decomp_attn": None,
        },
    },
    "vae": {
        "module_name": "compiled_vae",
        "keywords": ["vae"],
        "dest_type": "numpy",
        "export_fn": vae.export_vae_model,
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
    if isinstance(hf_model_name, dict):
        name = hf_model_name["text_encoder"]
    else:
        name = hf_model_name
    if name in [
        "stabilityai/sdxl-turbo",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "/models/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe",
        "/models/SDXL/official_pytorch/fp16/stable_diffusion_fp16//checkpoint_pipe",
    ]:
        return sdxl_model_map
    elif "stabilityai/stable-diffusion-3" in name:
        return sd3_model_map
    else:
        return sd1_sd2_model_map


torch_dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "float32": torch.float32,
    "float16": torch.float16,
    "int8": torch.int8,
    "i8": torch.int8,
}


class SharkSDPipeline(TurbinePipelineBase):
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
        self.model_max_length = max_length
        self.height = height
        self.width = width
        self.cpu_scheduling = cpu_scheduling
        self.scheduler_id = scheduler_id
        self.num_inference_steps = num_inference_steps
        self.punet_quant_paths = punet_quant_paths

        self.text_encoder = None
        self.unet = None
        self.mmdit = None
        self.vae = None
        self.scheduler = None

        self.split_scheduler = True
        self.add_tk_kernels = add_tk_kernels
        self.tk_kernels_dir = tk_kernels_dir

        self.base_model_name = (
            hf_model_name
            if isinstance(hf_model_name, str)
            else hf_model_name.get("unet", hf_model_name.get("mmdit"))
        )
        self.is_img2img = False
        self.is_sdxl = "xl" in self.base_model_name.lower()
        self.is_sd3 = "stable-diffusion-3" in self.base_model_name
        if self.is_sdxl:
            if self.split_scheduler:
                if self.map.get("unetloop"):
                    self.map.pop("unetloop")
                if self.map.get("fullpipeline"):
                    self.map.pop("fullpipeline")
            self.tokenizers = [
                CLIPTokenizer.from_pretrained(
                    self.base_model_name, subfolder="tokenizer"
                ),
                CLIPTokenizer.from_pretrained(
                    self.base_model_name, subfolder="tokenizer_2"
                ),
            ]
            self.map["text_encoder"]["export_args"]["batch_input"] = batch_prompts
            self.diffusion_model = self.map["unet"]
            if vae_weight_path is not None:
                self.map["vae"]["export_args"]["external_weight_path"] = vae_weight_path
            self.map["vae"]["export_args"]["vae_harness"] = vae_harness
        elif self.is_sd3:
            self.tokenizer = SD3Tokenizer()
            self.scheduler_id = "EulerFlowDiscrete"
            self.map["text_encoder"]["export_args"]["external_weights"] = "irpa"
            self.map["text_encoder"]["export_args"][
                "external_weight_path"
            ] = "stable_diffusion_3_medium_text_encoder_fp16.irpa"
            self.diffusion_model = self.map["mmdit"]
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.base_model_name, subfolder="tokenizer"
            )
            self.diffusion_model = self.map["unet"]

        self.latents_precision = self.diffusion_model["precision"]
        self.latents_channels = self.map["vae"]["export_args"]["num_channels"]
        self.scheduler_device = self.diffusion_model["device"]
        self.scheduler_driver = self.diffusion_model["driver"]
        self.scheduler_target = self.diffusion_model["target"]

        self.latents_dtype = torch_dtypes[self.latents_precision]
        self.use_i8_punet = self.use_punet = use_i8_punet
        if self.use_punet:
            self.setup_punet()
        elif not self.is_sd3:
            self.map["unet"]["keywords"].append("!punet")
            self.map["unet"]["function_name"] = "run_forward"

    def setup_punet(self):
        if self.use_i8_punet:
            if self.add_tk_kernels:
                self.map["unet"]["export_args"]["add_tk_kernels"] = self.add_tk_kernels
                self.map["unet"]["export_args"]["tk_kernels_dir"] = self.tk_kernels_dir
            self.map["unet"]["export_args"]["precision"] = "i8"
            self.map["unet"]["export_args"]["external_weight_path"] = (
                utils.create_safe_name(self.base_model_name) + "_punet_dataset_i8.irpa"
            )
            self.map["unet"]["export_args"]["quant_paths"] = self.punet_quant_paths
            for idx, word in enumerate(self.map["unet"]["keywords"]):
                if word in ["fp32", "fp16"]:
                    self.map["unet"]["keywords"][idx] = "i8"
                    break
        self.map["unet"]["export_args"]["use_punet"] = True
        self.map["unet"]["use_weights_for_export"] = True
        self.map["unet"]["keywords"].append("punet")
        self.map["unet"]["module_name"] = "compiled_punet"
        self.map["unet"]["function_name"] = "main"

    # LOAD

    def load_scheduler(
        self,
        scheduler_id: str = None,
        steps: int = 30,
    ):
        if not self.cpu_scheduling:
            if self.is_sd3:
                export_fn = sd3_schedulers.export_scheduler_model
            else:
                export_fn = scheduler.export_scheduler_model
            self.map["scheduler"] = {
                "module_name": "compiled_scheduler",
                "export_fn": export_fn,
                "driver": self.scheduler_driver,
                "export_args": {
                    "hf_model_name": self.base_model_name,
                    "scheduler_id": scheduler_id,
                    "batch_size": self.batch_size,
                    "height": self.height,
                    "width": self.width,
                    "num_inference_steps": steps,
                    "precision": self.latents_precision,
                    "compile_to": "vmfb",
                    "device": self.scheduler_device,
                    "target": self.scheduler_target,
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
            try:
                self.load_submodel("scheduler")
            except:
                print("JIT export of scheduler failed. Loading CPU scheduler.")
                self.cpu_scheduling = True
        if self.cpu_scheduling:
            if self.is_sd3:
                raise AssertionError("CPU scheduling not yet supported for SD3")
            else:
                scheduler_device = self.unet.device
            scheduler = schedulers.get_scheduler(
                self.base_model_name, self.scheduler_id
            )
            self.scheduler = schedulers.SharkSchedulerCPUWrapper(
                scheduler,
                self.batch_size,
                scheduler_device,
                latents_dtype=self.latents_dtype,
            )
            if self.use_punet:
                self.scheduler.use_punet = True

    # RUN

    def encode_prompts_sdxl(self, prompt, negative_prompt):
        # Tokenize prompt and negative prompt.
        text_input_ids_list = []
        uncond_input_ids_list = []

        for tokenizer in self.tokenizers:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=self.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids_list += text_inputs.input_ids.unsqueeze(0)
            uncond_input_ids_list += uncond_input.input_ids.unsqueeze(0)

        if self.compiled_pipeline:
            return text_input_ids_list, uncond_input_ids_list
        else:
            prompt_embeds, add_text_embeds = self.text_encoder(
                "encode_prompts", [*text_input_ids_list, *uncond_input_ids_list]
            )
            return prompt_embeds, add_text_embeds

    def encode_prompts_sd3(self, prompt, negative_prompt):
        text_input_ids_dict = self.tokenizer.tokenize_with_weights(prompt)
        uncond_input_ids_dict = self.tokenizer.tokenize_with_weights(negative_prompt)
        text_input_ids_list = list(text_input_ids_dict.values())
        uncond_input_ids_list = list(uncond_input_ids_dict.values())
        text_encoders_inputs = [
            text_input_ids_list[0],
            text_input_ids_list[1],
            text_input_ids_list[2],
            uncond_input_ids_list[0],
            uncond_input_ids_list[1],
            uncond_input_ids_list[2],
        ]
        return self.text_encoder("encode_tokens", text_encoders_inputs)

    def prepare_latents(
        self,
        noise,
        num_inference_steps,
        image=None,
        strength=None,
    ):
        if self.is_img2img:
            raise NotImplementedError("Image-to-image not supported yet.")
        elif self.is_sdxl and self.cpu_scheduling:
            self.scheduler.do_guidance = False
            self.scheduler.repeat_sample = False
            (
                sample,
                add_time_ids,
                step_indexes,
                timesteps,
            ) = self.scheduler.initialize_sdxl(noise, num_inference_steps)
            return sample, add_time_ids, step_indexes, timesteps
        elif self.is_sdxl or self.is_sd3:
            return self.scheduler("run_initialize", noise)
        else:
            sample, timesteps = self.scheduler.initialize_sd(noise, num_inference_steps)
            return sample, timesteps

    def get_rand_latents(self, seed, batch_count):
        samples = []
        uint32_info = np.iinfo(np.uint32)
        uint32_min, uint32_max = uint32_info.min, uint32_info.max
        if seed < uint32_min or seed >= uint32_max:
            seed = randint(uint32_min, uint32_max)
        for i in range(batch_count):
            generator = torch.manual_seed(seed + i)
            rand_sample = torch.randn(
                (
                    self.batch_size,
                    self.latents_channels,
                    self.height // 8,
                    self.width // 8,
                ),
                generator=generator,
                dtype=self.latents_dtype,
            )
            samples.append(rand_sample)
        return samples

    def _produce_latents_sd(
        self,
        sample,
        prompt_embeds,
        negative_prompt_embeds,
        steps,
        guidance_scale,
    ):
        image = None
        strength = 0
        sample, timesteps = self.prepare_latents(
            sample, self.num_inference_steps, image, strength
        )
        text_embeddings = torch.cat((negative_prompt_embeds, prompt_embeds), dim=0)
        self.scheduler.do_guidance = False
        for i, t in tqdm(enumerate(timesteps)):
            latent_model_input, _ = self.scheduler.scale_model_input(sample, t)
            timestep = torch.tensor([t])
            unet_inputs = [
                latent_model_input,
                timestep,
            ]
            unet_inputs.extend([text_embeddings, [guidance_scale]])
            latents = self.unet(self.map["unet"]["function_name"], unet_inputs)
            sample = self.scheduler.step(
                torch.tensor(
                    latents, dtype=torch_dtypes[self.map["unet"]["precision"]]
                ),
                t,
                sample,
            )
        return sample

    def _produce_latents_sdxl(
        self,
        sample,
        prompt_embeds,
        add_text_embeds,
        steps,
        guidance_scale,
    ):
        image = None
        strength = 0
        latents, add_time_ids, step_indexes, timesteps = self.prepare_latents(
            sample, self.num_inference_steps, image, strength
        )
        guidance_scale = ireert.asdevicearray(
            self.unet.device,
            [guidance_scale],
            dtype=self.map["unet"]["np_dtype"],
        )
        # Disable progress bar if we aren't in verbose mode or if we're printing
        # benchmark latencies for unet.
        for i, t in tqdm(
            enumerate(timesteps),
            disable=(self.map["unet"].get("benchmark") or not self.verbose),
        ):
            if self.cpu_scheduling:
                latent_model_input, t = self.scheduler.scale_model_input(
                    latents,
                    t,
                )
                t = t.type(self.map["unet"]["torch_dtype"])
            else:
                step = torch.tensor([i], dtype=torch.float32)
                latent_model_input, t = self.scheduler(
                    "run_scale", [latents, step, timesteps]
                )
            unet_inputs = [
                latent_model_input,
                t,
                prompt_embeds,
                add_text_embeds,
                add_time_ids,
                guidance_scale,
            ]
            if self.use_punet:
                for inp_idx, inp in enumerate(unet_inputs):
                    if not isinstance(inp, ireert.DeviceArray):
                        unet_inputs[inp_idx] = ireert.asdevicearray(
                            self.unet.device, inp, dtype=self.map["unet"]["np_dtype"]
                        )
            noise_pred = self.unet(
                self.map["unet"]["function_name"],
                unet_inputs,
            )
            if self.cpu_scheduling:
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                )
            else:
                latents = self.scheduler("run_step", [noise_pred, t, latents])
        return latents

    def _produce_latents_sd3(
        self,
        sample,
        prompt_embeds,
        pooled_prompt_embeds,
        steps,
        guidance_scale,
    ):
        image = None
        strength = 0
        latents, steps, timesteps = self.scheduler(
            "run_initialize",
            sample,
        )
        guidance_scale = ireert.asdevicearray(
            self.mmdit.device,
            [guidance_scale],
            dtype=self.map["mmdit"]["np_dtype"],
        )
        # Disable progress bar if we aren't in verbose mode or if we're printing
        # benchmark latencies for unet.
        for i, t in tqdm(
            enumerate(timesteps),
            disable=(self.map["mmdit"].get("benchmark") or not self.verbose),
        ):
            step = torch.tensor([i], dtype=torch.float32)
            latent_model_input, t = self.scheduler(
                "run_scale", [latents, step, timesteps]
            )
            mmdit_inputs = [
                latent_model_input,
                prompt_embeds,
                pooled_prompt_embeds,
                t,
            ]
            noise_pred = self.mmdit(
                "run_forward",
                mmdit_inputs,
            )
            latents = self.scheduler(
                "run_step", [noise_pred, t, latents, guidance_scale]
            )
        return latents

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

        samples = self.get_rand_latents(seed, batch_count)

        # Tokenize prompt and negative prompt.
        if self.is_sdxl:
            prompt_embeds, negative_embeds = self.encode_prompts_sdxl(
                prompt, negative_prompt
            )
        elif self.is_sd3:
            prompt_embeds, negative_embeds = self.encode_prompts_sd3(
                prompt, negative_prompt
            )
        else:
            prompt_embeds, negative_embeds = encode_prompt(
                self, prompt, negative_prompt
            )

        for i in range(batch_count):
            produce_latents_input = [
                samples[i],
                prompt_embeds,
                negative_embeds,
                steps,
                guidance_scale,
            ]
            if self.is_sdxl:
                latents = self._produce_latents_sdxl(*produce_latents_input)
            elif self.is_sd3:
                latents = self._produce_latents_sd3(*produce_latents_input)
            else:
                latents = self._produce_latents_sd(*produce_latents_input)
            image = self.vae("decode", [latents])
            numpy_images.append(image)
            pipe_end = time.time()

        logging.info(f"Total inference time: {pipe_end - pipe_start:.2f}s")
        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        images = []
        for idx, image in enumerate(numpy_images):
            if self.is_sd3:
                if image.ndim == 4:
                    image = image[0]
                image = torch.from_numpy(image).cpu().permute(1, 2, 0).float().numpy()
                image = (image * 255).round().astype("uint8")
                out_image = Image.fromarray(image)
                images.extend([out_image])
            else:
                image = (
                    torch.from_numpy(image).cpu().permute(0, 2, 3, 1).float().numpy()
                )
                image = numpy_to_pil_image(image)
                images.append(image[0])
        if return_imgs:
            return images
        for idx, image in enumerate(images):
            img_path = "sd_output_" + timestamp + "_" + str(idx) + ".png"
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
        "unet": args.ireec_flags + args.unet_flags,
        "mmdit": args.ireec_flags + args.mmdit_flags,
        "vae_decode": args.ireec_flags + args.vae_flags,
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
        "unet": (args.unet_decomp_attn if args.unet_decomp_attn else args.decomp_attn),
        "mmdit": (
            args.mmdit_decomp_attn if args.mmdit_decomp_attn else args.decomp_attn
        ),
        "vae": args.vae_decomp_attn if args.vae_decomp_attn else args.decomp_attn,
    }
    sd_pipe = SharkSDPipeline(
        args.hf_model_name,
        args.height,
        args.width,
        args.batch_size,
        args.max_length,
        args.precision,
        args.device,
        args.iree_target_triple,
        ireec_flags,
        args.attn_spec,
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
    sd_pipe.prepare_all()
    sd_pipe.load_map()
    sd_pipe.generate_images(
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
