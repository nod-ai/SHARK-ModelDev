# Copyright 2024 Advanced Micro Devices, inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import torch
import ast
from collections.abc import Iterable

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

np_dtypes = {
    "fp16": np.float16,
    "fp32": np.float32,
    "float16": np.float16,
    "float32": np.float32,
}
torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "float16": torch.float16,
    "float32": torch.float32,
}


def merge_arg_into_map(model_map, arg, arg_name):
    if isinstance(arg, dict):
        for key in arg.keys():
            if key not in model_map.keys():
                continue
            if not model_map[key].get(arg_name):
                model_map[key][arg_name] = arg[key]
    else:
        for key in model_map.keys():
            if not model_map[key].get(arg_name):
                model_map[key][arg_name] = arg
    return model_map


def merge_export_arg(model_map, arg, arg_name):
    if isinstance(arg, dict):
        for key in arg.keys():
            if key not in model_map.keys():
                continue
            if arg_name not in model_map[key].get("export_args", {}):
                model_map[key]["export_args"][arg_name] = arg[key]
    else:
        for key in model_map.keys():
            if not model_map[key].get("export_args", {}).get(arg_name):
                continue
            model_map[key]["export_args"][arg_name] = arg
    return model_map


# def str_to_list(string):
#     out = string.strip("[]").replace(" ", "").split(";")
#     for item in out:
#         item = ast.literal_eval(item)
#     return out


class PipelineComponent:
    """
    Wraps a VMFB runner with attributes for embedded metadata, device info, utilities and
    has methods for handling I/O or otherwise assisting in interfacing with their pipeline
    and its other components.
    This aims to make new pipelines and execution modes easier to write, manage, and debug.
    """

    def __init__(
        self, printer, dest_type="devicearray", dest_dtype="float16", benchmark=False
    ):
        self.runner = None
        self.module_name = None
        self.device = None
        self.metadata = None
        self.printer = printer
        self.benchmark = benchmark
        self.dest_type = dest_type
        self.dest_dtype = dest_dtype

    def load(
        self,
        rt_device: str,
        vmfb_path: str,
        module_name: str,
        external_weight_path: str = None,
        extra_plugin=None,
    ):
        self.module_name = module_name
        self.printer.print(
            f"Loading {module_name} from {vmfb_path} with external weights: {external_weight_path}."
        )
        self.runner = vmfbRunner(
            rt_device, vmfb_path, external_weight_path, extra_plugin
        )
        self.device = self.runner.config.device
        self.module = getattr(self.runner.ctx.modules, module_name)
        self.get_metadata()

    def unload(self):
        self.device = None
        self.runner = None
        gc.collect()

    def get_metadata(self):
        self.metadata = {}
        for function_name in self.module.vm_module.function_names:
            if any(x in function_name for x in ["$async", "__init"]):
                continue
            try:
                self.metadata[function_name] = self.module[
                    function_name
                ].vm_function.reflection
            except:
                logging.warning(
                    f"Could not get metadata for {self.module_name}['{function_name}']."
                )
                self.metadata[function_name] = None

    def _validate_or_convert_inputs(self, function_name, inputs):
        val_inputs = [None for i in inputs]
        if self.metadata.get(function_name):
            expected_input_shapes = self.metadata.get(function_name, {}).get(
                "input_shapes"
            )
            if expected_input_shapes:
                expected_input_shapes = ast.literal_eval(expected_input_shapes)
            expected_input_dtypes = self.metadata.get(function_name, {}).get(
                "input_dtypes", ""
            )
            if expected_input_dtypes:
                expected_input_dtypes = ast.literal_eval(expected_input_dtypes)
            if not expected_input_dtypes:
                pass
            if not expected_input_shapes:
                logging.warning(
                    f"No input shapes found for {self.module_name}['{function_name}']."
                )
                for idx, i in enumerate(inputs):
                    if not isinstance(i, ireert.DeviceArray):
                        val_inputs[idx] = ireert.asdevicearray(self.device, i)
                pass
            if not isinstance(expected_input_shapes, list):
                expected_input_shapes = [expected_input_shapes]
            for i, input_dtype in enumerate(expected_input_dtypes):
                if not isinstance(inputs[i], ireert.DeviceArray):
                    val_inputs[i] = ireert.asdevicearray(
                        self.device, inputs[i], input_dtype
                    )
                elif str(inputs[i].dtype).split(".")[-1] != input_dtype:
                    logging.warning(
                        f"Converting input {i} to {input_dtype} for {self.module_name}['{function_name}']."
                    )
                    val_inputs[i] = inputs[i].astype(input_dtype)
                else:
                    val_inputs[i] = inputs[i]
            for i, input_shape in enumerate(expected_input_shapes):
                if isinstance(input_shape, str):
                    input_shape = ast.literal_eval(input_shape)
                elif not input_shape:
                    continue
                actual = tuple(val_inputs[i].shape)
                expected = tuple(input_shape)
                for idx, shape in enumerate(expected):
                    if shape == "?":
                        pass
                    elif actual[idx] != shape:
                        raise ValueError(
                            f"Expected input {i} to be of shape {input_shape} for {self.module_name}['{function_name}'], got {str(tuple(inputs[i].shape))}."
                        )
        else:
            for idx, i in enumerate(inputs):
                if not isinstance(i, ireert.DeviceArray):
                    val_inputs[idx] = ireert.asdevicearray(self.device, i)
                else:
                    val_inputs[idx] = inputs[idx]
        return val_inputs

    def _output_cast(self, output):
        if isinstance(output, tuple):
            out_tuple = ()
            for array in output:
                array_out = self._output_cast(array)
                out_tuple += (array_out,)
            return out_tuple
        match self.dest_type:
            case "devicearray":
                output = (
                    output.astype(self.dest_dtype)
                    if output.dtype != self.dest_dtype
                    else output
                )
                return output
            case "torch":
                output = torch.tensor(
                    output.to_host(), dtype=torch_dtypes[self.dest_dtype]
                )
                return output
            case "numpy":
                return output.to_host().astype(np_dtypes[self.dest_dtype])
            case _:
                return output

    def _run(self, function_name, inputs: list):
        return self.module[function_name](*inputs)

    def _run_and_benchmark(self, function_name, inputs: list):
        start_time = time.time()
        output = self._run(function_name, inputs)
        latency = time.time() - start_time
        self.printer.print(
            f"Latency for {self.module_name}['{function_name}']: {latency}sec"
        )
        return output

    def __call__(self, function_name, inputs: list):
        casted_output = False
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = self._validate_or_convert_inputs(function_name, inputs)
        if self.benchmark:
            output = self._run_and_benchmark(function_name, inputs)
        else:
            output = self._run(function_name, inputs)
        output = self._output_cast(output)
        return output


class Printer:
    def __init__(self, verbose, start_time, print_time):
        """
        verbose: 0 for silence, 1 for print
        start_time: time of construction (or reset) of this Printer
        last_print: time of last call to 'print' method
        print_time: 1 to print with time prefix, 0 to not
        """
        self.verbose = verbose
        self.start_time = start_time
        self.last_print = start_time
        self.print_time = print_time

    def reset(self):
        if self.print_time:
            if self.verbose:
                self.print("Will now reset clock for printer to 0.0 [s].")
            self.last_print = time.time()
            self.start_time = time.time()
            if self.verbose:
                self.print("Clock for printer reset to t = 0.0 [s].")

    def print(self, message):
        if self.verbose:
            # Print something like "[t=0.123 dt=0.004] 'message'"
            if self.print_time:
                time_now = time.time()
                print(
                    f"[t={time_now - self.start_time:.3f} dt={time_now - self.last_print:.3f}] {message}"
                )
                self.last_print = time_now
            else:
                print(f"{message}")


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
    target: str | dict[str]
        Either a string i.e. "gfx1100", or a dictionary with keys matching the submodels of a given pipeline.
    ireec_flags: str | dict[str]
        A comma-separated string of flags to pass to the IREE compiler, or a dict of them with keys matching submodels of a given pipeline.
    """

    def __init__(
        self,
        model_map: dict,
        device: str | dict[str],
        target: str | dict[str],
        ireec_flags: str | dict[str] = None,
        precision: str | dict[str] = "fp16",
        td_spec: str | dict[str] = None,
        decomp_attn: bool | dict[bool] = False,
        external_weights: str | dict[str] = None,
        pipeline_dir: str = "./shark_vmfbs",
        external_weights_dir: str = "./shark_weights",
        hf_model_name: str | dict[str] = None,
        benchmark: bool | dict[bool] = False,
        verbose: bool = False,
        common_export_args: dict = {},
    ):
        self.map = model_map
        self.printer = Printer(verbose, time.time(), True)
        if isinstance(device, dict):
            assert isinstance(
                target, dict
            ), "Device and target triple must be both dicts or both strings."
            for submodel in self.map.keys():
                assert submodel in device.keys(), f"Device for {submodel} not found."
                assert (
                    submodel in target.keys()
                ), f"Target arch for {submodel} not found."
                self.map[submodel]["device"] = utils.iree_backend_map(device[submodel])
                self.map[submodel]["driver"] = utils.iree_device_map(device[submodel])
                self.map[submodel]["target"] = target[submodel]
        else:
            assert isinstance(
                target, str
            ), "Device and target triple must be both dicts or both strings."
            for submodel in self.map.keys():
                self.map[submodel]["device"] = utils.iree_backend_map(device)
                self.map[submodel]["driver"] = utils.iree_device_map(device)
                self.map[submodel]["target"] = target

        map_arguments = {
            "ireec_flags": ireec_flags,
            "precision": precision,
            "td_spec": td_spec,
            "decomp_attn": decomp_attn,
            "external_weights": external_weights,
            "hf_model_name": hf_model_name,
            "benchmark": benchmark,
        }
        for arg in map_arguments.keys():
            self.map = merge_arg_into_map(self.map, map_arguments[arg], arg)

        self.map = merge_arg_into_map(
            self.map, np_dtypes[self.map[submodel]["precision"]], "np_dtype"
        )
        self.map = merge_arg_into_map(
            self.map, torch_dtypes[self.map[submodel]["precision"]], "torch_dtype"
        )
        for arg in common_export_args.keys():
            for submodel in self.map.keys():
                self.map[submodel].get("export_args", {})[arg] = self.map[submodel].get(
                    arg, common_export_args[arg]
                )
        for submodel in self.map.keys():
            for key, value in map_arguments.items():
                self.map = merge_export_arg(self.map, value, key)
            for key, value in self.map[submodel].get("export_args", {}).items():
                if key == "hf_model_name":
                    self.map[submodel]["keywords"].append(
                        utils.create_safe_name(value.split("/")[-1], "")
                    )
                if key == "decomp_attn":
                    if not value:
                        self.map[submodel]["keywords"].append("!decomp_attn")
                    else:
                        self.map[submodel]["keywords"].append("decomp_attn")
                elif key == "batch_size":
                    self.map[submodel]["keywords"].append(f"bs{value}")
                elif key in ["height"]:
                    dims = f"{self.map[submodel]['export_args']['width']}x{self.map[submodel]['export_args']['height']}"
                    self.map[submodel]["keywords"].append(dims)
                elif key in ["max_length", "precision"]:
                    self.map[submodel]["keywords"].append(str(value))

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
                self.printer.print("All necessary files found.")
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
                        self.printer.print("Fetching: ", submodel)
                        self.export_submodel(
                            submodel, input_mlir=self.map[submodel].get("mlir")
                        )
                        if not self.map[submodel]["export_args"]["external_weights"]:
                            assert not self.map[submodel].get(
                                "weights"
                            ), f"External weights should not be used for a model with inlined params."
                    if not self.map[submodel].get("weights") and self.map[submodel][
                        "export_args"
                    ].get("external_weights"):
                        self.export_submodel(submodel, weights_only=True)
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
            mlir_keywords = copy.deepcopy(keywords)
            mlir_keywords.extend(
                [
                    "mlir",
                ]
            )
            keywords.extend(
                [
                    "vmfb",
                    self.map[key]["target"],
                ]
            )
            neg_keywords = []
            for kw in keywords:
                if kw.startswith("!"):
                    neg_keywords.append(kw.strip("!"))
                    keywords.remove(kw)
                    mlir_keywords.remove(kw)
            avail_files = os.listdir(pipeline_dir)
            candidates = []
            for filename in avail_files:
                if all(str(x) in filename for x in keywords) and not any(
                    x in filename for x in neg_keywords
                ):
                    candidates.append(os.path.join(pipeline_dir, filename))
                if all(str(x) in filename for x in mlir_keywords) and not any(
                    x in filename for x in neg_keywords
                ):
                    self.map[key]["mlir"] = os.path.join(pipeline_dir, filename)
            if len(candidates) == 1:
                self.map[key]["vmfb"] = candidates[0]
            elif len(candidates) > 1:
                self.printer.print(f"Multiple files found for {key}: {candidates}")
                self.printer.print(f"Choosing {candidates[0]} for {key}.")
                self.map[key]["vmfb"] = candidates[0]
            else:
                # vmfb not found in pipeline_dir. Add to list of files to generate.
                missing[key].append("vmfb")

            # Make sure vmfb needs external weights, as they may be inlined.
            if self.map[key].get("export_args", {}).get("external_weights"):
                if not self.map[key]["external_weights"]:
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
                    self.map[key]["export_args"]["external_weight_path"],
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
                    self.printer.print(
                        f"Multiple weight files found for {key}: {candidates}"
                    )
                    self.printer.print(f"Choosing {candidates[0]} for {key}.")
                    self.map[key][weights] = candidates[0]
                elif self.map[key].get("external_weights"):
                    # weights not found in external_weights_dir. Add to list of files to generate.
                    missing[key].append("weights")
        if not any(x for x in missing.values()):
            ready = True
        else:
            self.printer.print("Missing files: ", missing)
            ready = False
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

        if (
            self.map[submodel].get("external_weights")
            and self.external_weights_dir
            and not self.map[submodel].get("weights")
        ):
            if not os.path.exists(self.external_weights_dir):
                os.makedirs(self.external_weights_dir, exist_ok=False)

            self.map[submodel]["export_args"]["external_weight_path"] = os.path.join(
                self.external_weights_dir,
                self.map[submodel]["export_args"]["external_weight_path"],
            )
        elif self.map[submodel].get("weights") and self.map[submodel].get(
            "use_weights_to_export"
        ):
            self.map[submodel]["export_args"]["external_weight_path"] = self.map[
                submodel
            ]["weights"]

        elif not self.map[submodel].get("external_weights"):
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
        self.map[submodel]["export_args"]["input_mlir"] = self.map[submodel].get(
            "mlir", input_mlir
        )

        match submodel:
            case "unetloop":  # SDXL ONLY FOR NOW
                pipeline_file = get_pipeline_ir(
                    self.map[submodel]["export_args"]["width"],
                    self.map[submodel]["export_args"]["height"],
                    self.map[submodel]["export_args"]["precision"],
                    self.map[submodel]["export_args"]["batch_size"],
                    self.map[submodel]["export_args"]["max_length"],
                    "unet_loop",
                )
                dims = [
                    self.map[submodel]["export_args"]["width"],
                    self.map[submodel]["export_args"]["height"],
                ]
                dims = "x".join([str(x) for x in dims])
                pipeline_keys = [
                    utils.create_safe_name(
                        self.map[submodel]["export_args"]["hf_model_name"].split("/")[
                            -1
                        ],
                        "",
                    ),
                    "bs" + str(self.map[submodel]["export_args"]["batch_size"]),
                    dims,
                    self.map[submodel]["export_args"]["precision"],
                    str(self.map[submodel]["export_args"]["max_length"]),
                    "unetloop",
                ]
                vmfb_path = utils.compile_to_vmfb(
                    pipeline_file,
                    self.map["unet"]["device"],
                    self.map["unet"]["target"],
                    None,
                    os.path.join(self.pipeline_dir, "_".join(pipeline_keys)),
                    return_path=True,
                    mlir_source="str",
                )
                self.map[submodel]["vmfb"] = vmfb_path
                self.map[submodel]["weights"] = None
            case "fullpipeline":  # SDXL ONLY FOR NOW
                pipeline_file = get_pipeline_ir(
                    self.map[submodel]["export_args"]["width"],
                    self.map[submodel]["export_args"]["height"],
                    self.map[submodel]["export_args"]["precision"],
                    self.map[submodel]["export_args"]["batch_size"],
                    self.map[submodel]["export_args"]["max_length"],
                    "tokens_to_image",
                )
                dims = [
                    self.map[submodel]["export_args"]["width"],
                    self.map[submodel]["export_args"]["height"],
                ]
                dims = "x".join([str(x) for x in dims])
                pipeline_keys = [
                    utils.create_safe_name(
                        self.map[submodel]["export_args"]["hf_model_name"].split("/")[
                            -1
                        ],
                        "",
                    ),
                    "bs" + str(self.map[submodel]["export_args"]["batch_size"]),
                    dims,
                    self.map[submodel]["export_args"]["precision"],
                    str(self.map[submodel]["export_args"]["max_length"]),
                    "fullpipeline",
                ]
                vmfb_path = utils.compile_to_vmfb(
                    pipeline_file,
                    self.map["unet"]["device"],
                    self.map["unet"]["target"],
                    None,
                    os.path.join(self.pipeline_dir, "_".join(pipeline_keys)),
                    return_path=True,
                    mlir_source="str",
                )
                self.map[submodel]["vmfb"] = vmfb_path
                self.map[submodel]["weights"] = None
            case _:
                export_args = self.map[submodel].get("export_args", {})
                if weights_only:
                    export_args["weights_only"] = True
                    export_args["input_mlir"] = None
                if export_args:
                    exported = self.map[submodel]["export_fn"](**export_args)
                else:
                    exported = self.map[submodel]["export_fn"]()
                if not self.map[submodel].get("weights") and self.map[submodel][
                    "export_args"
                ].get("external_weights", None):
                    self.map[submodel]["weights"] = self.map[submodel][
                        "export_args"
                    ].get("external_weight_path", None)
                if not weights_only:
                    self.map[submodel]["vmfb"] = exported

    # LOAD
    def load_map(self):
        for submodel in self.map.keys():
            if not self.map[submodel]["load"]:
                self.printer.print("Skipping load for ", submodel)
                continue
            self.load_submodel(submodel)

    def load_submodel(self, submodel):
        if not self.map[submodel].get("vmfb"):
            raise ValueError(f"VMFB not found for {submodel}.")
        if not self.map[submodel].get("weights") and self.map[submodel].get(
            "external_weights"
        ):
            raise ValueError(f"Weights not found for {submodel}.")
        dest_type = self.map[submodel].get("dest_type", "devicearray")
        self.map[submodel]["runner"] = PipelineComponent(
            printer=self.printer,
            dest_type=dest_type,
            benchmark=self.map[submodel].get("benchmark", False),
        )
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
