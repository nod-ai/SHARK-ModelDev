# Copyright 2024 Advanced Micro Devices, inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import torch
from tqdm.auto import tqdm
from turbine_models.custom_models.sd3_inference import (
    sd3_text_encoders,
    sd3_mmdit,
    sd3_vae,
    sd3_schedulers,
)
from turbine_models.custom_models.sd3_inference.text_encoder_impls import SD3Tokenizer
import iree.runtime as ireert
from turbine_models.custom_models.sd_inference import utils
from turbine_models.model_runner import vmfbRunner
from transformers import CLIPTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler

from PIL import Image
import os
import numpy as np
import time
import copy
from datetime import datetime as dt

empty_pipe_dict = {
    "clip": None,
    "mmdit": None,
    "scheduler": None,
    "vae": None,
}

EMPTY_FLAGS = {
    "clip": None,
    "mmdit": None,
    "vae": None,
    "pipeline": None,
}


class SharkSD3Pipeline:
    def __init__(
        self,
        hf_model_name: str,
        height: int,
        width: int,
        precision: str,
        max_length: int,
        batch_size: int,
        num_inference_steps: int,
        device: str | dict[str],
        iree_target_triple: str | dict[str],
        ireec_flags: dict = EMPTY_FLAGS,
        attn_spec: str = None,
        decomp_attn: bool = False,
        pipeline_dir: str = "./shark_vmfbs",
        external_weights_dir: str = "./shark_weights",
        external_weights: str = "safetensors",
        vae_decomp_attn: bool = False,
        cpu_scheduling: bool = False,
        vae_precision: str = "fp32",
        scheduler_id: str = None, #compatibility only, always uses EulerFlowScheduler
        shift: float = 1.0,

    ):
        self.hf_model_name = hf_model_name
        # self.scheduler_id = scheduler_id
        self.height = height
        self.width = width
        self.shift = shift
        self.precision = precision
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_inference_steps = None
        self.devices = {}
        if isinstance(device, dict):
            assert isinstance(
                iree_target_triple, dict
            ), "Device and target triple must be both dicts or both strings."
            self.devices["clip"] = {
                "device": device["clip"],
                "driver": utils.iree_device_map(device["clip"]),
                "target": iree_target_triple["clip"],
            }
            self.devices["mmdit"] = {
                "device": device["mmdit"],
                "driver": utils.iree_device_map(device["mmdit"]),
                "target": iree_target_triple["mmdit"],
            }
            self.devices["vae"] = {
                "device": device["vae"],
                "driver": utils.iree_device_map(device["vae"]),
                "target": iree_target_triple["vae"],
            }
        else:
            assert isinstance(
                iree_target_triple, str
            ), "Device and target triple must be both dicts or both strings."
            self.devices["clip"] = {
                "device": device,
                "driver": utils.iree_device_map(device),
                "target": iree_target_triple,
            }
            self.devices["mmdit"] = {
                "device": device,
                "driver": utils.iree_device_map(device),
                "target": iree_target_triple,
            }
            self.devices["vae"] = {
                "device": device,
                "driver": utils.iree_device_map(device),
                "target": iree_target_triple,
            }
        self.iree_target_triple = iree_target_triple
        self.ireec_flags = ireec_flags if ireec_flags else EMPTY_FLAGS
        self.attn_spec = attn_spec
        self.decomp_attn = decomp_attn
        self.pipeline_dir = pipeline_dir
        self.external_weights_dir = external_weights_dir
        self.external_weights = external_weights
        self.vae_decomp_attn = vae_decomp_attn
        self.custom_vae = None
        self.cpu_scheduling = cpu_scheduling
        self.torch_dtype = torch.float32 if self.precision == "fp32" else torch.float16
        self.vae_precision = vae_precision if vae_precision else self.precision
        self.vae_dtype = torch.float32 if vae_precision == "fp32" else torch.float16
        # TODO: set this based on user-inputted guidance scale and negative prompt.
        self.do_classifier_free_guidance = True  # False if any(x in hf_model_name for x in ["turbo", "lightning"]) else True
        self._interrupt = False
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
                        print("Fetching: ", submodel)
                        vmfb, weight = self.export_submodel(submodel, input_mlir=mlirs)
                        vmfbs[submodel] = vmfb
                        if weights[submodel] is None:
                            weights[submodel] = weight
                    elif weights[submodel] is None and not any(
                        x in submodel for x in ["pipeline", "scheduler"]
                    ):
                        _, weight = self.export_submodel(submodel, weights_only=True)
                        weights[submodel] = weight
                ready, vmfbs, weights = self.is_prepared(vmfbs, weights)
                if ready:
                    print("All necessary files found.")
                    return vmfbs, weights
                else:
                    print("There was an error generating the necessary files.")
                    exit()
        else:
            print("All necessary files found. Loading pipeline.")
        return vmfbs, weights

    def is_prepared(self, vmfbs, weights):
        missing = []
        height = self.height
        width = self.width
        for key in vmfbs:
            if key == "scheduler":
                continue
            elif key == "vae":
                keywords = ["vae", self.vae_precision, height, width]
                device_key = "vae"
            elif key == "clip":
                keywords = ["text_encoders", self.precision, self.max_length]
                device_key = "clip"
            else:
                keywords = [key, self.precision, self.max_length, height, width]
                device_key = key
            avail_files = os.listdir(self.pipeline_dir)
            keywords.append("vmfb")
            keywords.append(self.devices[device_key]["target"])
            for filename in avail_files:
                if all(str(x) in filename for x in keywords):
                    vmfbs[key] = os.path.join(self.pipeline_dir, filename)
            if not vmfbs[key]:
                missing.append(key + " vmfb")
        for w_key in weights:
            if any(x in w_key for x in ["pipeline", "scheduler"]):
                continue
            if weights[w_key] is not None:
                continue
            if self.external_weights is None:
                continue
            default_name = os.path.join(
                self.external_weights_dir, w_key + "." + self.external_weights
            )
            if w_key == "clip":
                default_name = os.path.join(
                    self.external_weights_dir, f"sd3_text_encoders_{self.precision}.irpa"
                )
            if w_key == "mmdit":
                default_name = os.path.join(
                    self.external_weights_dir,
                    f"sd3_mmdit_{self.precision}." + self.external_weights,
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
            case "vae":
                vae_torch = sd3_vae.VaeModel(
                    # This is a public model, so no auth required
                    self.hf_model_name,
                )
                return vae_torch
            case "mmdit":
                mmdit_torch = sd3_mmdit.MMDiTModel(
                    dtype=self.torch_dtype,
                )
                return mmdit_torch

    def export_submodel(
        self,
        submodel: str,
        input_mlir: str = None,
        weights_only: bool = False,
    ):
        if not os.path.exists(self.pipeline_dir):
            os.makedirs(self.pipeline_dir)
        if self.external_weights and self.external_weights_dir:
            if not os.path.exists(self.external_weights_dir):
                os.makedirs(self.external_weights_dir, exist_ok=True)
            vae_external_weight_path = os.path.join(
                self.external_weights_dir, f"sd3_vae_{self.vae_precision}." + self.external_weights
            )
            mmdit_external_weight_path = os.path.join(
                self.external_weights_dir,
                f"sd3_mmdit_{self.precision}." + self.external_weights,
            )
            text_encoders_external_weight_path = os.path.join(
                self.external_weights_dir, f"sd3_text_encoders_{self.precision}.irpa"
            )
        elif self.external_weights is None:
            print(
                "No external weights type specified using --external_weights, weights for imported .mlir files will not be externalized."
            )
            vae_external_weight_path = None
            mmdit_external_weight_path = None
            text_encoders_external_weight_path = None
        else:
            print(
                f"No external weights directory specified using --external_weights_dir, we assume you have your own weights in {self.pipeline_dir}."
            )
            if not os.path.exists(self.pipeline_dir):
                os.makedirs(self.pipeline_dir, exist_ok=True)
            vae_external_weight_path = os.path.join(
                self.pipeline_dir, f"sd3_vae_{self.vae_precision}." + self.external_weights
            )
            mmdit_external_weight_path = os.path.join(
                self.pipeline_dir,
                f"sd3_mmdit_{self.precision}." + self.external_weights,
            )
            text_encoders_external_weight_path = os.path.join(
                self.pipeline_dir, f"sd3_text_encoders_{self.precision}.irpa"
            )
        if weights_only:
            input_mlir = {
                "vae": None,
                "clip": None,
                "mmdit": None,
                "scheduler": None,
            }
        match submodel:
            case "mmdit":
                if not input_mlir[submodel]:
                    mmdit_torch = self.get_torch_models("mmdit")
                else:
                    mmdit_torch = None
                mmdit_vmfb = sd3_mmdit.export_mmdit_model(
                    mmdit_torch,
                    self.hf_model_name,
                    self.batch_size,
                    self.height,
                    self.width,
                    self.precision,
                    self.max_length,
                    None,
                    "vmfb",
                    self.external_weights,
                    mmdit_external_weight_path,
                    self.devices["mmdit"]["device"],
                    self.devices["mmdit"]["target"],
                    self.ireec_flags["mmdit"],
                    self.decomp_attn,
                    exit_on_vmfb=False,
                    pipeline_dir=self.pipeline_dir,
                    attn_spec=self.attn_spec,
                    input_mlir=input_mlir["mmdit"],
                    weights_only=weights_only,
                )
                del mmdit_torch
                return mmdit_vmfb, mmdit_external_weight_path
            case "scheduler":
                scheduler_vmfb = sd3_schedulers.export_scheduler_model(
                    self.hf_model_name,
                    self.batch_size,
                    self.height,
                    self.width,
                    self.shift,
                    self.num_inference_steps,
                    self.precision,
                    "vmfb",
                    self.devices["mmdit"]["device"],
                    self.devices["mmdit"]["target"],
                    self.ireec_flags["scheduler"],
                    exit_on_vmfb=False,
                    pipeline_dir=self.pipeline_dir,
                    input_mlir=None,
                )
                return scheduler_vmfb, None
            case "vae":
                if not input_mlir[submodel]:
                    vae_torch = self.get_torch_models("vae")
                else:
                    vae_torch = None
                vae_vmfb = sd3_vae.export_vae_model(
                    vae_torch,
                    self.hf_model_name,
                    self.batch_size,
                    self.height,
                    self.width,
                    self.vae_precision,
                    "vmfb",
                    self.external_weights,
                    vae_external_weight_path,
                    self.devices["vae"]["device"],
                    self.devices["vae"]["target"],
                    self.ireec_flags["vae"],
                    self.vae_decomp_attn,
                    exit_on_vmfb=False,
                    pipeline_dir=self.pipeline_dir,
                    attn_spec=self.attn_spec,
                    input_mlir=input_mlir["vae"],
                    weights_only=weights_only,
                )
                del vae_torch
                return vae_vmfb, vae_external_weight_path
            case "clip":
                _, text_encoders_vmfb = sd3_text_encoders.export_text_encoders(
                    self.hf_model_name,
                    None,
                    self.max_length,
                    self.precision,
                    "vmfb",
                    self.external_weights,
                    text_encoders_external_weight_path,
                    self.devices["clip"]["device"],
                    self.devices["clip"]["target"],
                    self.ireec_flags["clip"],
                    exit_on_vmfb=False,
                    pipeline_dir=self.pipeline_dir,
                    input_mlir=input_mlir["clip"],
                    attn_spec=self.attn_spec,
                    output_batchsize=self.batch_size,
                )
                return text_encoders_vmfb, text_encoders_external_weight_path

    # LOAD

    def load_pipeline(
        self,
        vmfbs: dict,
        weights: dict,
        compiled_pipeline: bool = False,
        split_scheduler: bool = True,
        extra_device_args: dict = {},
    ):
        if "npu_delegate_path" in extra_device_args.keys():
            delegate = extra_device_args["npu_delegate_path"]
        else:
            delegate = None

        self.runners = {}
        runners = {}
        load_start = time.time()
        runners["pipe"] = vmfbRunner(
            self.devices["mmdit"]["driver"],
            vmfbs["mmdit"],
            weights["mmdit"],
        )
        unet_loaded = time.time()
        print("\n[LOG] MMDiT loaded in ", unet_loaded - load_start, "sec")

        # if not self.cpu_scheduling:
        #     runners["scheduler"] = sd3_schedulers.SharkSchedulerWrapper(
        #         self.devices["mmdit"]["driver"],
        #         vmfbs["scheduler"],
        #     )
        # else:
        #     print("Using torch CPU scheduler.")
        #     runners["scheduler"] = FlowMatchEulerDiscreteScheduler.from_pretrained(
        #         self.hf_model_name, subfolder="scheduler"
        #     )

        # sched_loaded = time.time()
        # print("\n[LOG] Scheduler loaded in ", sched_loaded - unet_loaded, "sec")
        runners["vae"] = vmfbRunner(
            self.devices["vae"]["driver"],
            vmfbs["vae"],
            weights["vae"],
            extra_plugin=delegate,
        )
        vae_loaded = time.time()
        print("\n[LOG] VAE Decode loaded in ", vae_loaded - unet_loaded, "sec")
        runners["clip"] = vmfbRunner(
            self.devices["clip"]["driver"],
            vmfbs["clip"],
            weights["clip"],
        )
        clip_loaded = time.time()
        print("\n[LOG] Text Encoders loaded in ", clip_loaded - vae_loaded, "sec")

        tok_start = time.time()
        self.tokenizer = SD3Tokenizer()
        tok_loaded = time.time()
        print("\n[LOG] Tokenizers loaded in ", tok_loaded - tok_start, "sec")
        self.runners = runners
        self.compiled_pipeline = compiled_pipeline
        print("Successfully loaded pipeline.")

    # RUN

    def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        batch_count: int = 1,
        guidance_scale: float = 4,
        seed: float = -1,
        return_imgs: bool = False,
        steps: int = None,
        cpu_scheduling: bool = False,
        scheduler_id: str = None,
        progress=None,
    ):
        needs_new_scheduler = (steps and steps != self.num_inference_steps) or cpu_scheduling != self.cpu_scheduling
        self.cpu_scheduling = cpu_scheduling
        if steps:
            self.num_inference_steps = steps
        if steps and not self.cpu_scheduling and needs_new_scheduler:
            self.runners["scheduler"] = None
            self.num_inference_steps = steps
            scheduler_path = f"EulerFlowScheduler_{self.num_inference_steps}"
            if not os.path.exists(scheduler_path):
               scheduler_path, _ = self.export_submodel("scheduler") 
            try:
                self.runners["scheduler"] = sd3_schedulers.SharkSchedulerWrapper(
                    self.devices["mmdit"]["driver"],
                    scheduler_path,
                )
            except:
                print("JIT export of scheduler failed. Loading CPU scheduler.")
                self.cpu_scheduling = True
        if self.cpu_scheduling and needs_new_scheduler:
            self.runners["scheduler"] = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.hf_model_name, subfolder="scheduler"
            )

        # TODO: implement case where this is false e.g. in SDXL Turbo
        do_classifier_free_guidance = True

        # Workaround for turbo support (guidance_scale 0)
        if guidance_scale == 0:
            negative_prompt = prompt
            prompt = ""

        iree_dtype = "float32" if self.precision == "fp32" else "float16"
        torch_dtype = torch.float32 if self.precision == "fp32" else torch.float16

        samples = []
        numpy_images = []

        for i in range(batch_count):
            generator = torch.Generator().manual_seed(int(seed))
            shape = (
                self.batch_size,
                16,
                self.height // 8,
                self.width // 8,
            )
            rand_sample = torch.randn(
                shape,
                generator=generator,
                dtype=torch.float32,
                layout=torch.strided,
            )
            samples.append(
                ireert.asdevicearray(
                    self.runners["pipe"].config.device, rand_sample, dtype=iree_dtype
                )
            )

        if not self.cpu_scheduling:
            guidance_scale = ireert.asdevicearray(
                self.runners["pipe"].config.device,
                np.asarray([guidance_scale]),
                dtype=iree_dtype,
            )

        tokenize_start = time.time()
        text_input_ids_dict = self.tokenizer.tokenize_with_weights(prompt)
        uncond_input_ids_dict = self.tokenizer.tokenize_with_weights(negative_prompt)
        text_input_ids_list = list(text_input_ids_dict.values())
        uncond_input_ids_list = list(uncond_input_ids_dict.values())
        text_encoders_inputs = [
            ireert.asdevicearray(
                self.runners["clip"].config.device, text_input_ids_list[0]
            ),
            ireert.asdevicearray(
                self.runners["clip"].config.device, text_input_ids_list[1]
            ),
            ireert.asdevicearray(
                self.runners["clip"].config.device, text_input_ids_list[2]
            ),
            ireert.asdevicearray(
                self.runners["clip"].config.device, uncond_input_ids_list[0]
            ),
            ireert.asdevicearray(
                self.runners["clip"].config.device, uncond_input_ids_list[1]
            ),
            ireert.asdevicearray(
                self.runners["clip"].config.device, uncond_input_ids_list[2]
            ),
        ]

        # Tokenize prompt and negative prompt.
        encode_prompts_start = time.time()
        prompt_embeds, pooled_prompt_embeds = self.runners[
            "clip"
        ].ctx.modules.compiled_text_encoder["encode_tokens"](*text_encoders_inputs)
        encode_prompts_end = time.time()
        if self.cpu_scheduling:
            timesteps, num_inference_steps = sd3_schedulers.retrieve_timesteps(
                self.runners["scheduler"],
                num_inference_steps=steps, 
                timesteps=None,
            )
            steps = num_inference_steps


        for i in range(batch_count):
            if self._interrupt:
                self._interrupt = False
                return
            unet_start = time.time()
            if not self.cpu_scheduling:
                latents, steps, timesteps = self.runners["scheduler"].initialize(samples[i])
            else:
                latents = torch.tensor(samples[i].to_host(), dtype=self.torch_dtype)
            iree_inputs = [
                latents,
                ireert.asdevicearray(
                    self.runners["pipe"].config.device, prompt_embeds, dtype=iree_dtype
                ),
                ireert.asdevicearray(
                    self.runners["pipe"].config.device,
                    pooled_prompt_embeds,
                    dtype=iree_dtype,
                ),
                None,
            ]
            for s in tqdm(iterable=range(steps), desc=f"Inference steps ({steps}), batch {i+1}"):
                if self._interrupt:
                    self._interrupt = False
                    return
                # print(f"step {s}")
                if self.cpu_scheduling:
                    step_index = s
                    t = timesteps[s]
                    if self.do_classifier_free_guidance:
                        latent_model_input = torch.cat([latents] * 2)
                    timestep = ireert.asdevicearray(
                        self.runners["pipe"].config.device,
                        t.expand(latent_model_input.shape[0]),
                        dtype=iree_dtype,
                    )
                    latent_model_input = ireert.asdevicearray(
                        self.runners["pipe"].config.device,
                        latent_model_input,
                        dtype=iree_dtype,
                    )
                else:
                    step_index = ireert.asdevicearray(
                        self.runners["scheduler"].runner.config.device,
                        torch.tensor([s]),
                        "int64",
                    )
                    latent_model_input, timestep = self.runners["scheduler"].prep(
                        latents,
                        step_index,
                        timesteps,
                    )
                    t = ireert.asdevicearray(
                        self.runners["scheduler"].runner.config.device,
                        timestep.to_host()[0]
                    )
                noise_pred = self.runners["pipe"].ctx.modules.compiled_mmdit[
                    "run_forward"
                ](
                    latent_model_input,
                    iree_inputs[1],
                    iree_inputs[2],
                    timestep,
                )
                if not self.cpu_scheduling:
                    latents = self.runners["scheduler"].step(
                        noise_pred,
                        t,
                        latents,
                        guidance_scale,
                        step_index,
                    )
                else:
                    noise_pred = torch.tensor(noise_pred.to_host(), dtype=self.torch_dtype)
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = self.runners["scheduler"].step(
                        noise_pred,
                        t,
                        latents,
                        return_dict=False,
                    )[0]
            if isinstance(latents, torch.Tensor):
                latents = latents.type(self.vae_dtype)
                latents = ireert.asdevicearray(
                    self.runners["vae"].config.device,
                    latents,
                )
            else:
                vae_numpy_dtype = np.float32 if self.vae_precision == "fp32" else np.float16
                latents = latents.astype(vae_numpy_dtype)

            vae_start = time.time()
            vae_out = self.runners["vae"].ctx.modules.compiled_vae["decode"](latents)

            pipe_end = time.time()

            image = vae_out.to_host()

            numpy_images.extend([image])
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
                (encode_prompts_end - encode_prompts_start) + (pipe_end - unet_start),
                "sec\n",
            )
            end = time.time()
            print("Total CLIP time:", encode_prompts_end - encode_prompts_start, "sec")
            print("Total tokenize time:", encode_prompts_start - tokenize_start, "sec")
            if batch_count > 1:
                print(
                    f"Total inference time ({batch_count} batch(es)):",
                    end - encode_prompts_start,
                    "sec",
                )
        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        images = []
        for idx, image in enumerate(numpy_images):
            if image.ndim == 4:
                image = image[0]
            image = torch.from_numpy(image).cpu().permute(1, 2, 0).float().numpy()
            image = (image * 255).round().astype("uint8")
            out_image = Image.fromarray(image)
            images.extend([[out_image]])
        if return_imgs:
            return images
        for idx_batch, image_batch in enumerate(images):
            for idx, image in enumerate(image_batch):
                img_path = (
                    "sd3_output_"
                    + timestamp
                    + "_"
                    + str(idx_batch)
                    + "_"
                    + str(idx)
                    + ".png"
                )
                image.save(img_path)
                print(img_path, "saved")
        return


def run_diffusers_cpu(
    hf_model_name,
    prompt,
    negative_prompt,
    guidance_scale,
    seed,
    height,
    width,
    num_inference_steps,
):
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        hf_model_name, torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    generator = torch.Generator().manual_seed(int(seed))

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    image.save(f"diffusers_reference_output_{timestamp}.png")


if __name__ == "__main__":
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args

    if args.compare_vs_torch:
        run_diffusers_cpu(
            args.hf_model_name,
            args.prompt,
            args.negative_prompt,
            args.guidance_scale,
            args.seed,
            args.height,
            args.width,
            args.num_inference_steps,
        )
        exit()
    map = empty_pipe_dict
    mlirs = copy.deepcopy(map)
    vmfbs = copy.deepcopy(map)
    weights = copy.deepcopy(map)

    if any(x for x in [args.clip_device, args.mmdit_device, args.vae_device]):
        assert all(
            x for x in [args.clip_device, args.mmdit_device, args.vae_device]
        ), "Please specify device for all submodels or pass --device for all submodels."
        assert all(
            x for x in [args.clip_target, args.mmdit_target, args.vae_target]
        ), "Please specify target triple for all submodels or pass --iree_target_triple for all submodels."
        args.device = "hybrid"
        args.iree_target_triple = "_".join(
            [args.clip_target, args.mmdit_target, args.vae_target]
        )
    else:
        args.clip_device = args.device
        args.mmdit_device = args.device
        args.vae_device = args.device
        args.clip_target = args.iree_target_triple
        args.mmdit_target = args.iree_target_triple
        args.vae_target = args.iree_target_triple

    devices = {
        "clip": args.clip_device,
        "mmdit": args.mmdit_device,
        "vae": args.vae_device,
    }
    targets = {
        "clip": args.clip_target,
        "mmdit": args.mmdit_target,
        "vae": args.vae_target,
    }
    ireec_flags = {
        "clip": args.ireec_flags + args.clip_flags,
        "mmdit": args.ireec_flags + args.unet_flags,
        "vae": args.ireec_flags + args.vae_flags,
        "pipeline": args.ireec_flags,
        "scheduler": args.ireec_flags,
    }
    if not args.pipeline_dir:
        pipe_id_list = [
            args.hf_model_name.split("/")[-1],
            str(args.height),
            str(args.width),
            str(args.max_length),
            args.precision,
            args.device,
            args.iree_target_triple,
        ]
        if args.decomp_attn:
            pipe_id_list.append("decomp")
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
    sd3_pipe = SharkSD3Pipeline(
        args.hf_model_name,
        args.height,
        args.width,
        args.precision,
        args.max_length,
        args.batch_size,
        args.num_inference_steps,
        devices,
        targets,
        ireec_flags,
        args.attn_spec,
        args.decomp_attn,
        args.pipeline_dir,
        args.external_weights_dir,
        external_weights=args.external_weights,
        vae_decomp_attn=args.vae_decomp_attn,
        cpu_scheduling=args.cpu_scheduling,
        vae_precision=args.vae_precision,
    )
    if args.cpu_scheduling:
        vmfbs.pop("scheduler")
        weights.pop("scheduler")
    vmfbs, weights = sd3_pipe.check_prepared(mlirs, vmfbs, weights)
    if args.npu_delegate_path:
        extra_device_args = {"npu_delegate_path": args.npu_delegate_path}
    else:
        extra_device_args = {}
    sd3_pipe.load_pipeline(
        vmfbs,
        weights,
        args.compiled_pipeline,
        args.split_scheduler,
        extra_device_args=extra_device_args,
    )
    sd3_pipe.generate_images(
        args.prompt,
        args.negative_prompt,
        args.batch_count,
        args.guidance_scale,
        args.seed,
        False,
    )
    print("Image generation complete.")
