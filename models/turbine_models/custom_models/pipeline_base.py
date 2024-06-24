# Copyright 2024 Advanced Micro Devices, inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import torch

import iree.runtime as ireert
from turbine_models.custom_models.sd_inference import utils, schedulers
from turbine_models.custom_models.sdxl_inference.pipeline_ir import (
    get_pipeline_ir,
)
from turbine_models.utils.sdxl_benchmark import run_benchmark
from turbine_models.model_runner import vmfbRunner

from PIL import Image
import gc
import os
import numpy as np
import time
import copy
from datetime import datetime as dt


def merge_arg_into_map(model_map, arg, arg_name):
    if isinstance(arg, dict):
        for key in arg.keys():
            if not model_map[key].get(arg_name):
                model_map[key][arg_name] = arg[key]
    else:
        for key in model_map.keys():
            if not model_map[key].get(arg_name):
                model_map[key][arg_name] = arg
    return model_map


class PipelineComponent:
    """
    Wraps a VMFB runner with attributes for embedded metadata, device info, utilities and
    has methods for handling I/O or otherwise assisting in interfacing with their pipeline
    and its other components.
    This aims to make new pipelines and execution modes easier to write, manage, and debug.
    """

    def __init__(self, dest_type=ireert.DeviceArray, dest_dtype="float16"):
        self.runner = None
        self.module_name = None
        self.device = None
        self.metadata = None
        self.benchmark = False
        self.output_type = dest_type
        self.output_dtype = dest_dtype

    def load(
        self,
        rt_device: str,
        vmfb_path: str,
        module_name: str,
        external_weight_path: str = None,
        extra_plugin=None,
    ):
        self.module_name = module_name
        self.runner = vmfbRunner(
            rt_device, vmfb_path, external_weight_path, extra_plugin
        )
        self.device = self.runner.config.device
        self.module = getattr(self.runner.ctx.modules, module_name)
        self.metadata = None

    def unload(self):
        self.device = None
        self.runner = None
        gc.collect()

    def get_metadata(self, function_name):
        if not self.metadata:
            self.metadata = self.module[function_name].vm_function.reflection
        return self.metadata

    def _run(self, function_name, inputs: list):
        print(inputs)
        return self.module[function_name](*inputs)

    def _run_and_benchmark(self, function_name, inputs: list):
        start_time = time.time()
        output = self._run(function_name, inputs)
        latency = time.time() - start_time
        print(f"Latency for {self.module_name}['{function_name}']: {latency}sec")
        return output

    def __call__(self, function_name, inputs: list):
        casted_output = False
        if not isinstance(inputs, list):
            inputs = [inputs]
        if self.benchmark:
            output = self._run_and_benchmark(function_name, inputs)
        else:
            output = self._run(function_name, inputs)
        if output.dtype != self.output_dtype:
            casted_output = True
            output = output.astype(self.output_dtype)
        match self.output_type:
            case ireert.DeviceArray:
                if casted_output:
                    output = ireert.asdevicearray(
                        self.device, output, self.output_dtype
                    )
                return output
            case torch.Tensor:
                return torch.tensor(output.to_host())
            case np.ndarray:
                return output.to_host()


class TurbinePipelineBase:
    """
    This class is a lightweight base for Stable Diffusion
    inference API classes. It should provide methods for:

    - Exporting and compiling a set (model map) of torch IR modules
    - preparing weights for an inference job
    - loading weights for an inference job
    - utilities i.e. filenames, downloads

    The general flow of an arbitrary child of this pipeline base is as follows:
    1. Initialize a model map and class attributes.
    2. Preparation: Check if all necessary files are present, and generate them if not. (prepare_all() / prepare_submodel())
        - This is done by submodel, so that users can generate a new submodel with the same pipeline.
        - If vmfb not found, first check turbine tank for matching .vmfb file.
        - If vmfb not downloadable, try downloading .mlir.
        - If neither on Azure, run the export function in model map to export to torch IR and compile with IREE.
        - If weights not found, run the export function in model map with weights_only=True.
            - Apps should populate the weights with custom weights by now so they can be managed and converted if needed here.
    3. Load the pipeline: Load the prepared files onto devices as vmfbRunners. (load_pipeline() / load_submodel() / reload_submodel())
    4. Run Inference:



    Arguments:
    model_map: dict
        A dictionary mapping submodel names to their export functions and hf model ids. This is used throughout the pipeline.
        It also should provide I/O information for the submodels.
    height: int
        The height of the image to be generated
    width: int
        The width of the image to be generated
    precision: str
        The precision of the image latents. This usually decides the precision of all models in the pipeline.
    max_length: int
        The maximum sequence length for text encoders and diffusion models.
    batch_size: int
        The number of images to generate from each inference batch. This changes the shapes in all submodels.
    device: str | dict[str]
        Either a string i.e. "rocm://0", or a dictionary of such with keys matching the submodels of a given pipeline.
        If a string, a dictionary will be created based on the pipeline's model map and the same device will be used for all submodels.
    iree_target_triple: str | dict[str]
        Either a string i.e. "gfx1100", or a dictionary with keys matching the submodels of a given pipeline.
    ireec_flags: str | dict[str]
        A comma-separated string of flags to pass to the IREE compiler, or a dict of them with keys matching submodels of a given pipeline.
    """

    def __init__(
        self,
        model_map: dict,
        batch_size: int,
        device: str | dict[str],
        iree_target_triple: str | dict[str],
        ireec_flags: str | dict[str] = None,
        precision: str | dict[str] = "fp16",
        td_spec: str | dict[str] = None,
        decomp_attn: bool | dict[bool] = False,
        external_weights: str | dict[str] = None,
        pipeline_dir: str = "./shark_vmfbs",
        external_weights_dir: str = "./shark_weights",
    ):
        self.map = model_map
        self.batch_size = batch_size
        if isinstance(device, dict):
            assert isinstance(
                iree_target_triple, dict
            ), "Device and target triple must be both dicts or both strings."
            for submodel in self.map.keys():
                assert submodel in device.keys(), f"Device for {submodel} not found."
                assert (
                    submodel in iree_target_triple.keys()
                ), f"Target arch for {submodel} not found."
                self.map[submodel]["device"] = device[submodel]
                self.map[submodel]["driver"] = utils.iree_device_map(device[submodel])
                self.map[submodel]["target"] = iree_target_triple[submodel]
        else:
            assert isinstance(
                iree_target_triple, str
            ), "Device and target triple must be both dicts or both strings."
            for submodel in self.map.keys():
                self.map[submodel]["device"] = device
                self.map[submodel]["driver"] = utils.iree_device_map(device)
                self.map[submodel]["target"] = iree_target_triple
        map_arguments = {
            "ireec_flags": ireec_flags,
            "precision": precision,
            "td_spec": td_spec,
            "decomp_attn": decomp_attn,
            "external_weights": external_weights,
        }
        for arg in map_arguments.keys():
            self.map = merge_arg_into_map(self.map, map_arguments[arg], arg)
        np_dtypes = {
            "fp16": np.float16,
            "fp32": np.float32,
        }
        torch_dtypes = {
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        for submodel in self.map.keys():
            self.map = merge_arg_into_map(
                self.map, np_dtypes[self.map[submodel]["precision"]], "np_dtype"
            )
            self.map = merge_arg_into_map(
                self.map, torch_dtypes[self.map[submodel]["precision"]], "torch_dtype"
            )
        print(self.map)

        self.pipeline_dir = pipeline_dir
        if not os.path.exists(self.pipeline_dir):
            os.makedirs(self.pipeline_dir)
        self.external_weights_dir = external_weights_dir
        if not os.path.exists(self.external_weights_dir):
            os.makedirs(self.external_weights_dir)

        # Disabled for now -- enable through option when turbine tank is ready.
        self.download = False

        # These arguments are set at run or load time.
        self.compiled_pipeline = False
        self.split_scheduler = False
        self.cpu_scheduling = False

        # TODO: set this based on user-inputted guidance scale and negative prompt.
        self.do_classifier_free_guidance = True  # False if any(x in hf_model_name for x in ["turbo", "lightning"]) else True
        self._interrupt = False

    # FILE MANAGEMENT AND PIPELINE SETUP

    def prepare_all(
        self,
        mlirs: dict = {},
        vmfbs: dict = {},
        weights: dict = {},
        interactive: bool = False,
    ):
        ready = self.is_prepared(vmfbs, weights)
        match ready:
            case True:
                print("All necessary files found.")
                return
            case False:
                if interactive:
                    do_continue = input(
                        f"\nIt seems you are missing some necessary files. Would you like to generate them now? (y/n)"
                    )
                    if do_continue.lower() != "y":
                        exit()
                for submodel in self.map.keys():
                    if not self.map[submodel].get("vmfb"):
                        print("Fetching: ", submodel)
                        self.export_submodel(submodel, input_mlir=mlirs)
                        if not self.map[submodel]["external_weights"]:
                            assert not self.map[submodel].get(
                                "weights"
                            ), f"External weights should not be used for a model with inlined params."
                return self.prepare_all(mlirs, vmfbs, weights, interactive)

    def is_prepared(self, vmfbs, weights):
        missing = {}
        ready = False
        pipeline_dir = self.pipeline_dir
        for key in self.map:
            missing[key] = []
            # vmfb is already present in model map
            if self.map[key].get("vmfb"):
                continue
            # vmfb is passed in to this function
            elif vmfbs.get(key):
                self.map[key]["vmfb"] = vmfbs[key]
                continue
            # search self.pipeline_dir for key-specific vmfb
            keywords = self.map[key].get("keywords", [])
            keywords.extend(
                [
                    self.map[key]["safe_name"],
                    "vmfb",
                    "bs" + str(self.batch_size),
                    self.map[key]["target"],
                    self.map[key]["precision"],
                ]
            )
            avail_files = os.listdir(pipeline_dir)
            candidates = []
            for filename in avail_files:
                if all(str(x) in filename for x in keywords):
                    candidates.append(os.path.join(pipeline_dir, filename))
            if len(candidates) == 1:
                self.map[key]["vmfb"] = candidates[0]
            elif len(candidates) > 1:
                print(f"Multiple files found for {key}: {candidates}")
                print(f"Choosing {candidates[0]} for {key}.")
                self.map[key]["vmfb"] = candidates[0]
            else:
                # vmfb not found in pipeline_dir. Add to list of files to generate.
                missing[key].append("vmfb")

            # Make sure vmfb needs external weights, as they may be inlined.
            if self.map[key].get("external_weights"):
                if self.map[key]["external_weights"]:
                    continue
                if self.map[key].get("weights"):
                    # weights already found in model map
                    continue
                elif weights.get(key):
                    # weights passed in to this function
                    self.map[key]["weights"] = weights[key]
                    continue
                # search self.external_weights_dir for key-specific weights
                w_keywords = [
                    self.map[key]["safe_name"],
                    self.map[key]["precision"],
                    self.map[key]["external_weights"],
                ]
                avail_files = os.listdir(self.external_weights_dir)
                candidates = []
                for filename in avail_files:
                    if all(str(x) in filename for x in w_keywords):
                        candidates.append(
                            os.path.join(self.external_weights_dir, filename)
                        )
                if len(candidates) == 1:
                    self.map[key]["weights"] = candidates[0]
                elif len(candidates) > 1:
                    print(f"Multiple weight files found for {key}: {candidates}")
                    print(f"Choosing {candidates[0]} for {key}.")
                    self.map[key][weights] = candidates[0]
                else:
                    # weights not found in external_weights_dir. Add to list of files to generate.
                    missing[key].append("weights")
        if not any(x for x in missing.values()):
            ready = True
        else:
            print("Missing files: ", missing)
        return ready

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

    def export_submodel(
        self,
        submodel: str,
        input_mlir: str = None,
        weights_only: bool = False,
    ):
        if not os.path.exists(self.pipeline_dir):
            os.makedirs(self.pipeline_dir)

        if self.map[submodel]["external_weights"] and self.external_weights_dir:
            if not os.path.exists(self.external_weights_dir):
                os.makedirs(self.external_weights_dir, exist_ok=False)

            self.map[submodel]["weights"] = os.path.join(
                self.external_weights_dir,
                f"{submodel}_{self.map[submodel]['precision']}."
                + self.map[submodel]["external_weights"],
            )

        elif not self.map[submodel].get("external_weights"):
            print(
                "No external weights type specified using --external_weights, weights for imported .mlir files will not be externalized."
            )
            self.map[submodel]["weights"] = None

        if weights_only:
            input_mlir = None
        elif "mlir" in self.map[submodel].keys():
            input_mlir = self.map[submodel]["mlir"]
        elif self.download:
            try:
                input_mlir = self.get_mlir_from_turbine_tank(
                    submodel, self.tank_container
                )
            except:
                input_mlir = None
        else:
            input_mlir = None
        self.map[submodel]["mlir"] = input_mlir

        match submodel:
            case "unetloop":  # SDXL ONLY FOR NOW
                pipeline_file = get_pipeline_ir(
                    self.width,
                    self.height,
                    self.precision,
                    self.batch_size,
                    self.max_length,
                    "unet_loop",
                )
                pipeline_keys = [
                    utils.create_safe_name(self.hf_model_name.split("/")[-1], ""),
                    "bs" + str(self.batch_size),
                    f"{str(self.width)}x{str(self.height)}",
                    self.precision,
                    str(self.max_length),
                    "unetloop",
                ]
                vmfb_path = utils.compile_to_vmfb(
                    pipeline_file,
                    self.map["unet"]["device"],
                    self.map["unet"]["target"],
                    self.ireec_flags["pipeline"],
                    os.path.join(self.pipeline_dir, "_".join(pipeline_keys)),
                    return_path=True,
                    mlir_source="str",
                )
                self.map[submodel]["vmfb"] = vmfb_path
                self.map[submodel]["weights"] = None
            case "fullpipeline":  # SDXL ONLY FOR NOW
                pipeline_file = get_pipeline_ir(
                    self.width,
                    self.height,
                    self.precision,
                    self.batch_size,
                    self.max_length,
                    "tokens_to_image",
                )
                pipeline_keys = [
                    utils.create_safe_name(self.hf_model_name.split("/")[-1], ""),
                    "bs" + str(self.batch_size),
                    f"{str(self.width)}x{str(self.height)}",
                    self.precision,
                    str(self.max_length),
                    "fullpipeline",
                ]
                vmfb_path = utils.compile_to_vmfb(
                    pipeline_file,
                    self.map["unet"]["device"],
                    self.map["unet"]["target"],
                    self.ireec_flags["pipeline"],
                    os.path.join(self.pipeline_dir, "_".join(pipeline_keys)),
                    return_path=True,
                    mlir_source="str",
                )
                self.map[submodel]["vmfb"] = vmfb_path
                self.map[submodel]["weights"] = None
            case _:
                export_args = self.map[submodel].get("export_args", {})
                if self.map[submodel].get("input_mlir"):
                    export_args["input_mlir"] = self.map[submodel].get("mlir")
                if export_args:
                    vmfb_path = self.map[submodel]["export_fn"](**export_args)
                else:
                    vmfb_path = self.map[submodel]["export_fn"]()

    # LOAD
    def load_map(self):
        for submodel in self.map.keys():
            self.load_submodel(submodel)

    def load_submodel(self, submodel):
        if not self.map[submodel].get("vmfb"):
            raise ValueError(f"VMFB not found for {submodel}.")
        if not self.map[submodel].get("weights") and self.map[submodel].get(
            "external_weights"
        ):
            raise ValueError(f"Weights not found for {submodel}.")
        self.map[submodel]["runner"] = PipelineComponent()
        self.map[submodel]["runner"].load(
            self.map[submodel]["driver"],
            self.map[submodel]["vmfb"],
            self.map[submodel]["module_name"],
            self.map[submodel].get("weights"),
            self.map[submodel].get("extra_plugin"),
        )
        setattr(self, submodel, self.map[submodel]["runner"])

    def unload_submodel(self, submodel):
        self.map[submodel]["runner"].unload()
        setattr(self, submodel, None)


def numpy_to_pil_image(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = []
        for batched_image in images:
            for image in range(0, batched_image.size(dim=0)):
                pil_images.append(Image.fromarray(image.squeeze(), mode="L"))
    else:
        pil_images = []
        for image in images:
            pil_images.append(Image.fromarray(image))
    return pil_images
