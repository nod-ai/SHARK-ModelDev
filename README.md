# SHARK Turbine

This repo is Nod-AI's integration repository for various model bringup
activities and CI. In 2023 and early 2024, it played a different role
by being the place where FX/Dynamo based torch-mlir and IREE toolsets
were developed, including:

* [Torch-MLIR FxImporter](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/extras/fx_importer.py)
* [Torch-MLIR ONNX Importer](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/extras/onnx_importer.py)
* [Torch-MLIR's ONNX C Importer](https://github.com/llvm/torch-mlir/tree/main/projects/onnx_c_importer)
* [IREE Turbine](https://github.com/iree-org/iree-turbine)
* [Sharktank and Shortfin](https://github.com/nod-ai/sharktank)

As these have all found upstream homes, this repo is a bit bare. We will
continue to use it as a staging ground for things that don't have a
more defined spot and as a way to drive certain kinds of upstreaming
activities.


## Current Projects

### turbine-models

The `turbine-models` project (under models/) contains ports and adaptations
of various (mostly HF) models that we use in various ways.

### CI

Integration CI for a variety of projects is rooted in this repo.

