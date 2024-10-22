# SHARK-ModelDev

This is the AMD SHARK team's integration repository that connects inference tasks, such as Stable Diffusion, from their various source libraries to the IREE/SHARK ML acceleration and deployment framework.

In 2023 and early 2024, it played a different role
by being the place where FX/Dynamo based torch-mlir and IREE toolsets
were developed, including:

* [Torch-MLIR FxImporter](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/extras/fx_importer.py)
* [Torch-MLIR ONNX Importer](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/extras/onnx_importer.py)
* [Torch-MLIR's ONNX C Importer](https://github.com/llvm/torch-mlir/tree/main/projects/onnx_c_importer)
* [IREE Turbine](https://github.com/iree-org/iree-turbine)
* [Sharktank and Shortfin](https://github.com/nod-ai/sharktank)

As these have all found upstream homes, this repo is now responsible for:
 - Exporting (via iree-turbine tooling) and compiling (via iree-compile) torch modules from various sources, mostly huggingface libraries
 - Carrying alternative (maximally exported and compiled) implementations to complex inference tasks e.g. Stable Diffusion (1.5, 2.1, SDXL, SD3, Flux)

### turbine-models

The `turbine-models` project (under models/) contains ports and adaptations
of various (mostly HF) models that we use in various ways.

The only implementation in turbine-models that is currently in use is its stable diffusion exports and pipeline, which are slated for migration to https://github.com/nod-ai/SHARK-Platform for productionization -- it is to be temporarily preserved here for proof of concept, functionality, and performance. The current state of the turbine-models SD implementation is heavily based on Diffusers' StableDiffusionPipelines, with the harnessing and actual inference code rewritten to offload as much as possible from torch (cpu) to the IREE compiler and runtime. 

In the near future, the remaining purpose of turbine-models is to maintain and validate the " library import -> nn.module -> iree-turbine (dynamo) -> iree-compile " export and compilation stack, for a few different key model classes or suites.
The model validation and benchmarking of compiled artifacts is under migration to [SHARK-TestSuite](https://github.com/nod-ai/SHARK-TestSuite)

### CI / Tracking

A number of model support tasks are tracked in this repo's issues, and its CI is designed to be the origin point for relevant MLIR/VMFB artifacts used further downstream in test/benchmark/regression suites.
